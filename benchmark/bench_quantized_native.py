"""Measure native quantized operators with matched inputs and thread counts."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import torch
import tensorplay as tp
from tensorplay.ao.nn.quantized.packed import PackedWeight


def measure(fn, synchronize, repetitions):
    for _ in range(10):
        fn()
    samples=[]
    for _ in range(7):
        synchronize()
        started=time.perf_counter_ns()
        for _ in range(repetitions):
            fn()
        synchronize()
        samples.append((time.perf_counter_ns()-started)/1000/repetitions)
    return statistics.median(samples)


def code_pair(codes,scale,zp,device):
    raw=np.asarray(codes,dtype='uint8')
    return (tp._C._make_per_tensor_quantized_tensor(tp.tensor(raw,device=device),scale,zp),
            torch._make_per_tensor_quantized_tensor(torch.tensor(raw,device=device),scale,zp))


def run(device,repetitions):
    rng=np.random.default_rng(905)
    records=[]
    def synchronize():
        if device=='cuda':
            tp.cuda.synchronize()
            torch.cuda.synchronize()
    def record(name,shape,native,reference,kind='integer',tolerance=1):
        a,b=native(),reference()
        synchronize()
        if tuple(a.shape)!=tuple(b.shape):
            raise AssertionError(f'{name} output shapes differ')
        if a.is_quantized() and b.is_quantized:
            if a.q_scale()!=b.q_scale() or a.q_zero_point()!=b.q_zero_point():
                raise AssertionError(f'{name} output affine grids differ')
        ac=a.int_repr().cpu().numpy() if a.is_quantized() else a.float().cpu().numpy()
        bc=b.int_repr().cpu().numpy() if b.is_quantized else b.float().cpu().numpy()
        error=float(np.max(np.abs(ac.astype('float64')-bc.astype('float64')),initial=0))
        if error>tolerance:
            raise AssertionError(f'{name} output error {error} exceeds {tolerance}')
        first=measure(native,synchronize,repetitions)
        second=measure(reference,synchronize,repetitions)
        records.append(dict(operator=name,device=device,shape=list(shape),reference_backend=kind,
            native_us=first,reference_us=second,speedup=second/first,max_error=error))
    for size in (256,65536,1048576):
        x,r=code_pair(rng.integers(0,256,size=size,dtype='uint8'),.07,119,device)
        if device=='cpu':
            operations=[('relu',lambda:tp._C.quantized_relu(x),lambda:torch.relu(r)),
                ('tanh',lambda:tp._C.quantized_tanh(x,2/256,128),lambda:torch.tanh(r)),
                ('hardswish',lambda:tp._C.quantized_hardswish(x,.07,119),lambda:torch.ops.quantized.hardswish(r,.07,119)),
                ('add_scalar',lambda:tp._C.quantized_scalar(x,.21,0),lambda:torch.ops.quantized.add_scalar(r,.21))]
            for name,a,b in operations:
                record(name,(size,),a,b)
        else:
            record('relu',(size,),lambda:tp._C.quantized_relu(x),lambda:torch.relu(r))
    for m,n,k in ((1,512,1024),(32,256,512),(256,512,1024)):
        x,r=code_pair(rng.integers(0,127,size=(m,k),dtype='uint8'),.03,61,device)
        codes=rng.integers(-63,64,size=(n,k),dtype='int8')
        w=tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes,device=device),.02,-3)
        rw=torch._make_per_tensor_quantized_tensor(torch.tensor(codes),.02,-3)
        bias=rng.normal(0,.2,size=n).astype('float32')
        packed=PackedWeight(w,tp.tensor(bias,device=device))
        native=lambda:packed.run(x,.2,119)
        if device=='cpu':
            rp=torch.ops.quantized.linear_prepack(rw,torch.tensor(bias))
            reference=lambda:torch.ops.quantized.linear(r,rp,.2,119)
            kind='integer'
        else:
            rx,weight,rb=r.dequantize().cuda(),rw.dequantize().cuda(),torch.tensor(bias,device='cuda')
            reference=lambda:torch.quantize_per_tensor(torch.nn.functional.linear(rx,weight,rb),.2,119,torch.quint8)
            kind='floating'
        record('linear',(m,n,k),native,reference,kind)
    if device=='cpu':
        for bits in (2,4):
            dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
            rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
            for size in (65536,1048577):
                values=rng.uniform(-1,4,size=size).astype('float32')
                x,r=tp.tensor(values),torch.tensor(values)
                native=lambda:tp.quantize_per_tensor(x,.03,1,dtype)
                reference=lambda:torch.quantize_per_tensor(r,.03,1,rdtype)
                record('quantize_'+str(bits)+'bit',(size,),native,reference,tolerance=0)
                q,rq=native(),reference()
                record('dequantize_'+str(bits)+'bit',(size,),lambda:q.dequantize(),lambda:rq.dequantize(),tolerance=0)
        for channels,spatial in ((32,16),(128,56)):
            shape=(4,channels,spatial,spatial)
            x,r=code_pair(rng.integers(0,256,size=shape,dtype='uint8'),.03,119,device)
            x=x.contiguous(memory_format=tp.channels_last)
            r=r.contiguous(memory_format=torch.channels_last)
            parameters=[rng.normal(0,.2,size=channels).astype('float32') for _ in range(3)]
            parameters.append(rng.uniform(.1,2,size=channels).astype('float32'))
            p=[tp.tensor(v) for v in parameters]
            rp=[torch.tensor(v) for v in parameters]
            record('batch_norm2d',shape,
                lambda:tp.ops.quantized.batch_norm2d(x,*p,1e-5,.07,119),
                lambda:torch.ops.quantized.batch_norm2d(r,*rp,1e-5,.07,119))
        for m,n,k in ((1,256,1024),(32,256,1024)):
            codes=rng.integers(0,16,size=(n,k),dtype='int32')
            weight=tp._C._convert_weight_to_int4pack_for_cpu(tp.tensor(codes),2)
            rw=torch._convert_weight_to_int4pack_for_cpu(torch.tensor(codes),2)
            values=rng.normal(0,.2,size=(m,k)).astype('float32')
            parameters=rng.uniform(-.05,.05,size=(k//64,n,2)).astype('float32')
            for dtype in ('float32','float16','bfloat16'):
                x=tp.tensor(values,dtype=getattr(tp,dtype))
                r=torch.tensor(values,dtype=getattr(torch,dtype))
                p=tp.tensor(parameters,dtype=getattr(tp,dtype))
                rp=torch.tensor(parameters,dtype=getattr(torch,dtype))
                record('int4_mm_'+dtype,(m,n,k),
                    lambda:tp._C._weight_int4pack_mm_for_cpu(x,weight,64,p),
                    lambda:torch._weight_int4pack_mm_for_cpu(r,rw,64,rp),
                    tolerance={'float32':1e-4,'float16':2e-3,'bfloat16':2e-2}[dtype])
        for name,gates in (('lstm',4),('gru',3)):
            params,reference_params=[],[]
            width,features,layers=128,256,2
            for layer in range(layers):
                for size in (features if layer==0 else width,width):
                    codes=rng.integers(-32,33,size=(gates*width,size),dtype='int8')
                    w=tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes),.001,0)
                    rw=torch._make_per_tensor_quantized_tensor(torch.tensor(codes),.001,0)
                    bias=np.zeros(gates*width,dtype='float32')
                    params.append(tp.ops.quantized.linear_prepack(w,tp.tensor(bias)))
                    reference_params.append(torch.ops.quantized.linear_prepack(rw,torch.tensor(bias)))
            cells=[tp.ops.quantized.make_quantized_cell_params_dynamic(params[2*i],params[2*i+1],
                tp.zeros(gates*width),tp.zeros(gates*width),True) for i in range(layers)]
            rcells=[torch.ops.quantized.make_quantized_cell_params_dynamic(reference_params[2*i],reference_params[2*i+1],
                torch.zeros(gates*width),torch.zeros(gates*width),True) for i in range(layers)]
            values=rng.normal(0,.2,size=(16,4,features)).astype('float32')
            x,r=tp.tensor(values),torch.tensor(values)
            h,rh=tp.zeros((layers,4,width)),torch.zeros((layers,4,width))
            hx=[h,tp.zeros_like(h)] if name=='lstm' else h
            rhx=[rh,torch.zeros_like(rh)] if name=='lstm' else rh
            native_op=getattr(tp.ops.tp,'quantized_'+name).input
            reference_op=getattr(torch.ops.aten,'quantized_'+name).input
            record('dynamic_'+name,values.shape,
                lambda:native_op(x,hx,cells,True,layers,0.,False,False,False)[0],
                lambda:reference_op(r,rhx,rcells,True,layers,0.,False,False,False)[0],tolerance=2e-5)
    if device=='cuda':
        for m,n,k in ((1,512,1024),(32,256,512),(256,512,1024)):
            codes=rng.integers(-63,64,size=(m,k),dtype='int8')
            weights=rng.integers(-63,64,size=(n,k),dtype='int8')
            x=tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes,device=device),.03,0)
            r=torch._make_per_tensor_quantized_tensor(torch.tensor(codes,device=device),.03,0)
            w=tp._C._make_per_tensor_quantized_tensor(tp.tensor(weights,device=device),.02,0)
            rw=torch._make_per_tensor_quantized_tensor(torch.tensor(weights,device=device),.02,0)
            packed=PackedWeight(w)
            rp=torch.ops.quantized.linear_prepack(rw,None)
            record('linear_signed',(m,n,k),lambda:packed.run(x,.2,0),
                lambda:torch.ops.quantized.linear(r,rp,.2,0))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda','all'],default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repetitions',type=int,default=100)
    parser.add_argument('--output',type=Path,default=Path('benchmark/quantized_native_results.json'))
    args=parser.parse_args()
    tp.set_num_threads(args.threads);torch.set_num_threads(args.threads)
    devices=['cpu','cuda'] if args.device=='all' else [args.device]
    records=[item for device in devices for item in run(device,args.repetitions)]
    data=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        processor=platform.processor(),threads=args.threads,repetitions=args.repetitions,
        reference_version=torch.__version__,results=records)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
