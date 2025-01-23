from pathlib import Path

import pyrodigal
import pyrodigal_gv
from pyfastatools import Parser
from pyfastatools._parser import RecordIterator

GeneFinderT = pyrodigal.GeneFinder | pyrodigal_gv.ViralGeneFinder


class OrfFinder:
    _orf_finder: GeneFinderT

    def __init__(self, virus_mode: bool = False, **kwargs):
        kwargs["meta"] = kwargs.pop("meta", True)
        kwargs["mask"] = kwargs.pop("mask", True)

        if virus_mode:
            self._orf_finder = pyrodigal_gv.ViralGeneFinder(**kwargs)
        else:
            self._orf_finder = pyrodigal.GeneFinder(**kwargs)

    def parse(self, file: Path) -> RecordIterator:
        # clean the header of the FASTA file so that the prodigal format is correct
        return Parser(file).clean()

    def find_orfs(self, sequence: str) -> pyrodigal.Genes:
        return self._orf_finder.find_genes(sequence)

    def find_orfs_from_file(self, file: Path) -> dict[str, pyrodigal.Genes]:
        return {
            record.header.to_string(): self.find_orfs(record.seq) for record in self.parse(file)
        }
