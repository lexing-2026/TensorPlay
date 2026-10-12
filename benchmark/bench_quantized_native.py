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
        ac=a.int_repr().cpu().numpy() if a.is_quantized() else a.cpu().numpy()
        bc=b.int_repr().cpu().numpy() if b.is_quantized else b.cpu().numpy()
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
                ('tanh',lambda:tp._C.quantized_tanh(x,1/256,128),lambda:torch.tanh(r)),
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
