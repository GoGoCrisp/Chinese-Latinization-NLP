"""Lossless atomic-unit adapter around unmodified Hugging Face pair-frequency BPE.

A proxy is ONE Unicode scalar identifying a whole atom, never a syllable letter.
No normalizer/pretokenizer/decoder is installed on the proxy engine. A preceding
single ASCII separator is carried inside the following syllable atom; extra
whitespace is encoded literally. Token IDs alone reconstruct the exact input.
"""
import json
import re
from functools import lru_cache

from tokenizers import Tokenizer
from tokenizers.models import BPE



class Atomizer:
    def __init__(self, representation, inventory=()):
        self.representation = representation
        self.inventory = frozenset(inventory)

    @lru_cache(maxsize=65536)
    def chunk_atoms(self, chunk):
        if chunk in self.inventory:
            return ((chunk, True),)
        result = []
        # Punctuation/digits do not invalidate a neighbouring complete syllable.
        # Do not guess internal syllable boundaries in unspaced letter strings.
        for part in re.findall(r'(?:[^\W\d_]|[\u0300-\u036f])+|[\s\S]', chunk):
            if part in self.inventory:
                result.append((part, True))
            else:
                result.extend((char, False) for char in part)
        return tuple(result)

    def units(self, text):
        if self.representation == 'chinese_origin':
            return list(text)
        atoms = []
        for match in re.finditer(r'\s+|\S+', text):
            chunk = match.group()
            for unit, is_syllable in self.chunk_atoms(chunk):
                if is_syllable and atoms and atoms[-1] == ' ':
                    atoms[-1] = ' ' + unit
                else:
                    atoms.append(unit)
        return atoms

class AtomicBPE:
    def __init__(self, metadata, engine=None):
        self.metadata = metadata
        self.atomizer = Atomizer(metadata['representation'], metadata['inventory'])
        self.atoms = metadata['atoms']
        self.atom_to_proxy = {a: chr(0xF0000+i) for i,a in enumerate(self.atoms)}
        self.proxy_to_bytes = {chr(0xF0000+i):
            bytes([int(a[2:],16)]) if a.startswith('B:') else a[2:].encode('utf-8')
            for i,a in enumerate(self.atoms)}
        self.engine = engine or Tokenizer(BPE(vocab={chr(0xF0000+i):i for i in range(len(self.atoms))}, merges=[]))

    def proxy(self, text):
        out=[]
        fallback=0
        for unit in self.atomizer.units(text):
            p=self.atom_to_proxy.get('T:'+unit)
            if p is None:
                fallback += 1
                out.extend(self.atom_to_proxy[f'B:{b:02x}'] for b in unit.encode('utf-8'))
            else:
                out.append(p)
        return ''.join(out),fallback

    def encode(self,text):
        proxy,_=self.proxy(text)
        return self.engine.encode(proxy,add_special_tokens=False).ids

    @lru_cache(maxsize=65536)
    def token_bytes(self, token_id):
        return b''.join(self.proxy_to_bytes[c] for c in self.engine.id_to_token(token_id))

    def decode(self,ids):
        return b''.join(self.token_bytes(i) for i in ids).decode('utf-8')

    def display(self,ids):
        result=[]
        for i in ids:
            data=self.token_bytes(i)
            try: result.append(data.decode('utf-8'))
            except UnicodeDecodeError: result.append('<UTF8:'+data.hex()+'>')
        return result

    @classmethod
    def load(cls,directory):
        return cls(json.loads((directory/'atoms.json').read_text()),Tokenizer.from_file(str(directory/'tokenizer.json')))
