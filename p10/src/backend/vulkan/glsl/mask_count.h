/*
 * A Bool payload in linear storage holds one byte per element, four bytes to a
 * word, the low byte naming the first element.  These helpers walk that packing.
 */

// Counts the bytes of one mask word that name an element below `elems` and hold
// a nonzero value.  `first` is the element the word's low byte holds, so a word
// that runs past the payload only contributes the bytes still inside it.
int mask_word_count(uint word, int first, int elems) {
  int count = 0;
  for (int byte = 0; byte < 4; ++byte) {
    if (first + byte >= elems) {
      break;
    }
    if (((word >> uint(8 * byte)) & 0xffu) != 0u) {
      ++count;
    }
  }
  return count;
}

// Tests one byte of a mask word, in the element order the count walk uses.
bool mask_byte_set(uint word, int byte) {
  return ((word >> uint(8 * byte)) & 0xffu) != 0u;
}
