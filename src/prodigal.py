import argparse
import itertools as it
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional, TextIO, overload

import pyrodigal
import pyrodigal_gv
from pyfastatools import Parser
from pyrodigal._version import __version__ as pyrodigal_version
from tqdm import tqdm


FASTA_PARSER_MSG = "Using C++ implemented fastaparser"


FASTA_WIDTH = 75
OUTPUT_FASTA_SUFFICES = Literal[".faa", ".ffn"]

GeneFinderT = pyrodigal.GeneFinder | pyrodigal_gv.ViralGeneFinder
FutureGenes = Future[pyrodigal.Genes]
LOGGER = sys.stdout


@dataclass
class Args:
    input: Optional[list[Path]]
    input_dir: Optional[Path]
    outdir: Path
    max_cpus: int
    genes: bool
    virus_mode: bool
    extension: str
    allow_unordered: bool

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace):
        fields = {
            field.name: getattr(namespace, field.name)
            for field in dataclass_fields(cls)
        }

        return cls(**fields)

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            description=(
                f"Find ORFs from query genomes using pyrodigal v{pyrodigal_version}, "
                "the cythonized prodigal API"
            )
        )

        input_args = parser.add_mutually_exclusive_group(required=True)

        input_args.add_argument(
            "-i",
            "--input",
            nargs="+",
            metavar="FILE",
            type=Path,
            help="fasta file(s) of query genomes (can use unix wildcards)",
        )
        input_args.add_argument(
            "-d",
            "--input-dir",
            metavar="DIR",
            type=Path,
            help="directory of fasta files to process",
        )

        parser.add_argument(
            "-o",
            "--outdir",
            default=Path.cwd(),
            type=Path,
            metavar="DIR",
            help=("output directory (default: %(default)s)"),
        )
        parser.add_argument(
            "-c",
            "--max-cpus",
            type=int,
            metavar="INT",
            default=1,
            help=("maximum number of threads to use (default: %(default)s)"),
        )
        parser.add_argument(
            "--genes",
            action="store_true",
            help="use to also output the nucleotide genes .ffn file",
        )
        parser.add_argument(
            "--virus-mode",
            action="store_true",
            help="use pyrodigal-gv to activate the virus models (default: %(default)s)",
        )
        parser.add_argument(
            "-x",
            "--extension",
            metavar="STR",
            default="fna",
            help="genome FASTA file extension if using -d/--input-dir (default: %(default)s)",
        )
        parser.add_argument(
            "--allow-unordered",
            action="store_true",
            help=(
                "for a single file input, this allows the protein ORFs to be written per scaffold "
                "as available. All protein ORFs for each scaffold will be in order, but the "
                "scaffolds will not necessarily be in the same order as in the input nucleotide "
                "file. **This is useful if you are extremely memory limited,** since the default "
                "strategy can lead to the ORFs being stored in memory for awhile before writing "
                "to file as the original scaffold order is maintained. NOTE: This is about 20 percent "
                "faster, so it is recommended to use this if the order of scaffolds does not "
                "matter."
            ),
        )
        return Args.from_namespace(parser.parse_args())


@overload
def create_orf_finder(virus_mode: Literal[False], **kwargs) -> pyrodigal.GeneFinder: ...


@overload
def create_orf_finder(
    virus_mode: Literal[True], **kwargs
) -> pyrodigal_gv.ViralGeneFinder: ...


@overload
def create_orf_finder(virus_mode: bool, **kwargs) -> GeneFinderT: ...


def create_orf_finder(virus_mode: bool, **kwargs) -> GeneFinderT:
    kwargs["meta"] = kwargs.pop("meta", True)
    kwargs["mask"] = kwargs.pop("mask", True)

    if virus_mode:
        return pyrodigal_gv.ViralGeneFinder(**kwargs)

    return pyrodigal.GeneFinder(**kwargs)


def get_output_name(file: Path, outdir: Path, suffix: OUTPUT_FASTA_SUFFICES) -> Path:
    return outdir.joinpath(file.with_suffix(suffix).name)


def _submit_sequences_to_pool(
    file: Path,
    pool: ThreadPoolExecutor,
    orf_finder: GeneFinderT,
    pbar: tqdm,
) -> dict[FutureGenes, str]:
    futures: dict[FutureGenes, str] = dict()
    for record in Parser(file):
        future = pool.submit(orf_finder.find_genes, record.seq)
        future.add_done_callback(lambda _: pbar.update())
        futures[future] = record.name

    return futures


def _process_single_future(
    future: FutureGenes,
    scaffold_name: str,
    protein_fp: TextIO,
    genes_fp: Optional[TextIO],
):
    scaffold_genes = future.result()
    scaffold_genes.write_translations(
        protein_fp, sequence_id=scaffold_name, width=FASTA_WIDTH
    )

    if genes_fp is not None:
        scaffold_genes.write_genes(
            genes_fp, sequence_id=scaffold_name, width=FASTA_WIDTH
        )

    # # delete the genes from memory
    del scaffold_genes
    future._result = None


def _unordered_future_processing(
    futures: dict[FutureGenes, str], protein_fp: TextIO, genes_fp: Optional[TextIO]
):

    # single thread writing to file
    for future in as_completed(futures):
        scaffold_name = futures[future]
        _process_single_future(future, scaffold_name, protein_fp, genes_fp)


def _ordered_future_processing(
    futures: dict[FutureGenes, str], protein_fp: TextIO, genes_fp: Optional[TextIO]
):
    completed = [False] * len(futures)

    start_idx = 0
    while not all(completed):
        current_futures = it.islice(futures.items(), start_idx, None)

        for i, (future, name) in enumerate(current_futures):
            real_idx = start_idx + i
            is_completed = completed[real_idx]

            # skip if already seen and written
            # kind of not necessary since we are already adjusting the starting point each time...
            if is_completed:
                continue

            if future.done():
                _process_single_future(future, name, protein_fp, genes_fp)

                completed[real_idx] = True
            else:
                # was not completed AND not done, so we need to
                # pick back up from here
                start_idx = real_idx
                break


def find_orfs_single_file(
    file: Path,
    orf_finder: GeneFinderT,
    outdir: Path,
    write_genes: bool,
    max_threads: int,
    allow_unordered: bool,
):
    protein_output = get_output_name(file, outdir, ".faa")
    genes_output = get_output_name(file, outdir, ".ffn")

    num_scaffolds = sum(1 for _ in Parser(file))
    n_threads = min(num_scaffolds, max_threads)

    with ExitStack() as ctx:
        pool = ctx.enter_context(ThreadPoolExecutor(max_workers=n_threads))
        protein_fp = ctx.enter_context(protein_output.open("w"))
        genes_fp = ctx.enter_context(genes_output.open("w")) if write_genes else None

        pbar = ctx.enter_context(
            tqdm(
                total=num_scaffolds,
                desc="Predicting ORFs for each scaffold",
                unit="scaffold",
                file=LOGGER,
            )
        )

        futures = _submit_sequences_to_pool(file, pool, orf_finder, pbar)

        if allow_unordered:
            _unordered_future_processing(futures, protein_fp, genes_fp)
        else:
            _ordered_future_processing(futures, protein_fp, genes_fp)


ScaffoldOrfs = dict[str, pyrodigal.Genes]
OutputT = tuple[Path, Path, bool, ScaffoldOrfs]
FindOrfsFnT = Callable[[Path, Path, bool, GeneFinderT], OutputT]


def _find_orfs(
    file: Path, outdir: Path, write_genes: bool, orf_finder: GeneFinderT
) -> OutputT:
    orfs = {record.name: orf_finder.find_genes(record.seq) for record in Parser(file)}

    return file, outdir, write_genes, orfs


def write_to_file(
    file: Path, scaffold_orfs: ScaffoldOrfs, outdir: Path, write_genes: bool
):
    protein_output = get_output_name(file, outdir, ".faa")
    genes_output = get_output_name(file, outdir, ".ffn")

    with ExitStack() as ctx:
        protein_fp = ctx.enter_context(protein_output.open("w"))
        genes_fp = ctx.enter_context(genes_output.open("w")) if write_genes else None

        for scaffold, orfs in scaffold_orfs.items():
            orfs.write_translations(protein_fp, sequence_id=scaffold, width=FASTA_WIDTH)

            if genes_fp is not None:
                orfs.write_genes(genes_fp, sequence_id=scaffold, width=FASTA_WIDTH)


def write_to_file_callback(future: Future[OutputT]):
    file, outdir, write_genes, scaffold_orfs = future.result()
    write_to_file(file, scaffold_orfs, outdir, write_genes)

    scaffold_orfs.clear()


def find_orfs_multiple_files(
    files: list[Path],
    outdir: Path,
    write_genes: bool,
    orf_finder: GeneFinderT,
    n_threads: int = 1,
):
    kwargs = dict(outdir=outdir, write_genes=write_genes, orf_finder=orf_finder)
    find_orfs_fn = partial(_find_orfs, **kwargs)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        pbar = tqdm(
            total=len(files),
            desc="Predicting ORFs for each file",
            unit="file",
            file=LOGGER,
        )

        futures: list[Future[OutputT]] = []
        for file in files:
            future = pool.submit(find_orfs_fn, file)
            future.add_done_callback(write_to_file_callback)
            future.add_done_callback(lambda f: pbar.update())
            futures.append(future)

        wait(futures)


def main():
    args = Args.parse_args()
    ext = args.extension
    if ext[0] != ".":
        ext = f".{ext}"

    if args.input_dir is not None:
        files = list(args.input_dir.glob(f"*{ext}"))
    elif args.input is not None:
        files = args.input
    else:
        raise ValueError("No input files provided")

    outdir = args.outdir
    write_genes = args.genes
    max_cpus = args.max_cpus

    outdir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        stream=LOGGER,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    msg = f"Predicting ORFs for {len(files)} files using Pyrodigal v{pyrodigal_version}"

    if args.virus_mode:
        msg += " with virus mode enabled (ie pyrodigal-gv)"

    logging.info(msg)

    orf_finder = create_orf_finder(virus_mode=args.virus_mode)

    # logging.info("Using C++ implemented fastaparser")
    logging.info(FASTA_PARSER_MSG)

    if len(files) == 1:
        # it is likely that this is a much larger than normal FASTA file
        # since viral genomes are typically single scaffolds, so they can all be in
        # a single file
        find_orfs_single_file(
            file=files[0],
            orf_finder=orf_finder,
            outdir=outdir,
            write_genes=write_genes,
            max_threads=max_cpus,
            allow_unordered=args.allow_unordered,
        )
    else:
        # this is likely for MAGs so each file is a single genome composed of multiple scaffolds
        n_threads = min(len(files), max_cpus)
        find_orfs_multiple_files(
            files=files,
            orf_finder=orf_finder,
            outdir=outdir,
            n_threads=n_threads,
            write_genes=write_genes,
        )

    logging.info(f"Finished predicting ORFs for {len(files)} file(s).")


if __name__ == "__main__":
    main()
