/* Section placement for the MACA device image.
 *
 * mxcc emits each device translation unit's image into an allocated,
 * read-only `.mc_fatbin` section.  The default script keeps allocated
 * read-only input in the segment that is laid out between the executable
 * `.text` segment and the writable `.got`/`.tbss` segment.  Once the
 * collected device image is large enough, that segment pushes `.got` and
 * `.tbss` more than 2^31 bytes away from `.text`, and the host code's
 * 32-bit PC-relative, GOT and TLS relocations no longer fit
 * ("relocation truncated to fit").
 *
 * Nothing in host code needs a 32-bit displacement into the device image:
 * the image is reached only through an absolute 64-bit pointer stored in
 * the small writable `.mcFatBinSegment` object.  Placing `.mc_fatbin` at
 * the end of the image therefore keeps `.text`, `.got` and `.tbss`
 * adjacent while leaving the device image exactly where the runtime can
 * still find it.
 *
 * Applied to the p10 link only when the MACA backend is enabled; for a
 * small device image the move is a no-op in effect.
 */
SECTIONS
{
  .mc_fatbin : { *(.mc_fatbin) }
}
INSERT AFTER .bss;
