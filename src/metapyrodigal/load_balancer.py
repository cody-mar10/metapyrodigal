from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack
from functools import partial
from itertools import islice
from pathlib import Path
from sys import stdout
from typing import Callable, Optional, TextIO

from pyfastatools import Parser
from pyrodigal import Genes
from tqdm import tqdm

from metapyrodigal.orf_finder import OrfFinder
from metapyrodigal.utils import get_output_name

FASTA_WIDTH = 75


class LoadBalancer(ABC):
    _pool: ThreadPoolExecutor

    def __init__(self, orf_finder: OrfFinder, n_threads: int = 1, progress_bar: bool = True):
        self.n_threads = n_threads
        self.orf_finder = orf_finder
        self.progress_bar = progress_bar
        self.futures: list[Future] = []

    def __enter__(self):
        self._pool = ThreadPoolExecutor(max_workers=self.n_threads)
        self.futures.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.shutdown(wait=True)
        del self._pool

    @property
    def pool(self) -> ThreadPoolExecutor:
        if not hasattr(self, "_pool"):
            raise RuntimeError(
                "LoadBalancer must be used as a context manager to create and teardown a thread pool."
            )
        return self._pool

    def _progress_bar(self, total: int, desc: str, unit: str) -> tqdm:
        return tqdm(total=total, desc=desc, unit=unit, file=stdout, disable=not self.progress_bar)

    def _submit(self, fn: Callable, *args):
        future = self.pool.submit(fn, *args)
        return future

    @abstractmethod
    def submit_to_pool(self, files: list[Path], outdir: Path, write_genes: bool): ...


class SingleFileLoadBalancer(LoadBalancer):
    """LoadBalancer for a single FASTA. For metagenomics, this would typically be an assembly file,
    a file of multiple single-scaffold genomes like viruses, or a single genome.

    This plans for the worst case of a single file with a large number of sequences, so it submits
    each sequence individually to the pool.
    """

    FutureGenes = Future[Genes]

    def __init__(
        self,
        orf_finder: OrfFinder,
        allow_unordered: bool = False,
        n_threads: int = 1,
        progress_bar: bool = True,
    ):
        super().__init__(orf_finder, n_threads, progress_bar)

        self.allow_unordered = allow_unordered

    def _process_single_future(
        self,
        future: FutureGenes,
        scaffold_name: str,
        protein_fp: TextIO,
        genes_fp: Optional[TextIO],
    ):
        scaffold_genes = future.result()
        scaffold_genes.write_translations(protein_fp, sequence_id=scaffold_name, width=FASTA_WIDTH)

        if genes_fp is not None:
            scaffold_genes.write_genes(genes_fp, sequence_id=scaffold_name, width=FASTA_WIDTH)

        # clear memory
        del scaffold_genes
        future._result = None

    def _unordered_processing(
        self, futures: dict[FutureGenes, str], protein_fp: TextIO, genes_fp: Optional[TextIO]
    ):
        for future in as_completed(futures):
            scaffold_name = futures[future]
            self._process_single_future(future, scaffold_name, protein_fp, genes_fp)

    def _ordered_processing(
        self, futures: dict[FutureGenes, str], protein_fp: TextIO, genes_fp: Optional[TextIO]
    ):
        completed = [False] * len(futures)

        start_idx = 0
        while not all(completed):
            current_futures = islice(futures.items(), start_idx, None)

            for i, (future, name) in enumerate(current_futures):
                real_idx = start_idx + i
                is_completed = completed[real_idx]

                # skip if already seen and written
                # kind of not necessary since we are already adjusting the starting point each time...
                if is_completed:
                    continue

                if future.done():
                    self._process_single_future(future, name, protein_fp, genes_fp)

                    completed[real_idx] = True
                else:
                    # was not completed AND not done, so we need to
                    # pick back up from here
                    start_idx = real_idx
                    break

    def _submit_sequences_to_pool(self, parser: Parser, pbar: tqdm) -> dict[FutureGenes, str]:
        futures: dict[SingleFileLoadBalancer.FutureGenes, str] = dict()
        for record in parser:
            future = self._submit(self.orf_finder.find_orfs, record.seq)
            future.add_done_callback(lambda _: pbar.update(1))
            futures[future] = record.header.name

        return futures

    def submit_to_pool(self, files: list[Path], outdir: Path, write_genes: bool):
        if len(files) != 1:
            raise ValueError("SingleFileLoadBalancer only accepts a single file.")

        file = files[0]

        protein_file = get_output_name(file, outdir, genes=False)
        genes_file = get_output_name(file, outdir, genes=True)

        parser = Parser(file)

        num_scaffolds = parser.num_records
        self.n_threads = min(self.n_threads, num_scaffolds)
        with ExitStack() as ctx:
            pbar = ctx.enter_context(
                self._progress_bar(
                    total=num_scaffolds, desc="Predicting ORFs for each scaffold", unit="scaffold"
                )
            )

            protein_fp = ctx.enter_context(protein_file.open("w"))
            genes_fp = ctx.enter_context(genes_file.open("w")) if write_genes else None

            futures = self._submit_sequences_to_pool(parser, pbar)

            if self.allow_unordered:
                # process as they are completed
                self._unordered_processing(futures, protein_fp, genes_fp)
            else:
                # process in order
                self._ordered_processing(futures, protein_fp, genes_fp)


class MultiFileLoadBalancer(LoadBalancer):
    """LoadBalancer for multiple FASTA files. For metagenomics, this would typically be from a
    directory containing multiple genomes or MAGs. However, this could also be multiple assemblies
    from different samples.

    The current plan is MAG-based in which we expect a large number of relatively small files.
    """

    # TODO: Need to plan for the case of a large number of large files. (Ie multiple assemblies)
    # One solution would be to submit individual scaffolds to the pool and then write as they come available

    # input file, output file, write_genes, genes
    FutureReturnType = tuple[Path, Path, bool, dict[str, Genes]]

    def _future_fn(self, file: Path, outdir: Path, write_genes: bool) -> FutureReturnType:
        genes = self.orf_finder.find_orfs_from_file(file)

        return file, outdir, write_genes, genes

    def _write(self, file: Path, outdir: Path, write_genes: bool, genes: dict[str, Genes]):
        protein_file = get_output_name(file, outdir, genes=False)
        genes_file = get_output_name(file, outdir, genes=True)

        with ExitStack() as ctx:
            protein_fp = ctx.enter_context(protein_file.open("w"))
            genes_fp = ctx.enter_context(genes_file.open("w")) if write_genes else None

            for scaffold, orfs in genes.items():
                orfs.write_translations(protein_fp, sequence_id=scaffold, width=FASTA_WIDTH)

                if genes_fp is not None:
                    orfs.write_genes(genes_fp, sequence_id=scaffold, width=FASTA_WIDTH)

    def _write_callback(self, future: Future[FutureReturnType]):
        file, outdir, write_genes, genes = future.result()

        self._write(file, outdir, write_genes, genes)

        # clear memory
        genes.clear()

    def submit_to_pool(self, files: list[Path], outdir: Path, write_genes: bool):
        if len(files) <= 1:
            raise ValueError("MultiFileLoadBalancer only accepts multiple files.")

        pbar = self._progress_bar(
            total=len(files), desc="Predicting ORFs for each file", unit="file"
        )

        future_fn = partial(self._future_fn, outdir=outdir, write_genes=write_genes)

        for file in files:
            future = self._submit(future_fn, file)

            future.add_done_callback(lambda _: pbar.update(1))
            future.add_done_callback(self._write_callback)

            self.futures.append(future)

        wait(self.futures)
        pbar.close()


def load_balancer(
    files: list[Path],
    orf_finder: OrfFinder,
    allow_unordered: bool,
    n_threads: int = 1,
    progress_bar: bool = True,
) -> LoadBalancer:
    if len(files) == 1:
        return SingleFileLoadBalancer(orf_finder, allow_unordered, n_threads, progress_bar)

    return MultiFileLoadBalancer(orf_finder, n_threads, progress_bar)
