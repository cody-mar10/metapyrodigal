import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from functools import partial
from pathlib import Path
from typing import Literal, Optional, overload

import pyrodigal
import pyrodigal_gv
from fastatools import FastaFile
from pyrodigal._version import __version__ as pyrodigal_version
from tqdm import tqdm

FASTA_WIDTH = 75
OUTPUT_FASTA_SUFFICES = Literal[".faa", ".ffn"]

GeneFinderT = pyrodigal.GeneFinder | pyrodigal_gv.ViralGeneFinder

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

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace):
        fields = {
            field.name: getattr(namespace, field.name)
            for field in dataclass_fields(cls)
        }

        return cls(**fields)


def parse_args() -> Args:
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
        help="use to also output the nucleotide genes .ffn file (default: %(default)s)",
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


def find_orfs_single_file(
    file: Path,
    orf_finder: GeneFinderT,
    outdir: Path,
    write_genes: bool,
    max_threads: int,
):
    protein_output = get_output_name(file, outdir, ".faa")
    genes_output = get_output_name(file, outdir, ".ffn")

    scaffolds = {
        record.header.name: record.sequence.encode()
        for record in FastaFile(file).parse()
    }

    n_threads = min(len(scaffolds), max_threads)

    with ExitStack() as ctx:
        pool = ctx.enter_context(ThreadPoolExecutor(max_workers=n_threads))
        protein_fp = ctx.enter_context(protein_output.open("w"))
        genes_fp = ctx.enter_context(genes_output.open("w")) if write_genes else None

        pbar = ctx.enter_context(
            tqdm(
                total=len(scaffolds),
                desc="Predicting ORFs for each scaffold",
                unit="scaffold",
                file=LOGGER,
            )
        )

        for scaffold_header, scaffold_genes in zip(
            scaffolds.keys(),
            pool.map(orf_finder.find_genes, scaffolds.values()),
        ):
            scaffold_genes.write_translations(
                protein_fp, sequence_id=scaffold_header, width=FASTA_WIDTH
            )

            if genes_fp is not None:
                scaffold_genes.write_genes(
                    genes_fp, sequence_id=scaffold_header, width=FASTA_WIDTH
                )

            pbar.update()


def find_orfs(
    file: Path,
    orf_finder: pyrodigal.GeneFinder,
) -> list[pyrodigal.Genes]:
    orfs = [
        orf_finder.find_genes(record.sequence) for record in FastaFile(file).parse()
    ]

    return orfs


def find_orfs_multiple_files(
    files: list[Path],
    orf_finder: GeneFinderT,
    outdir: Path,
    write_genes: bool,
    n_threads: int = 1,
):
    find_orfs_fn = partial(find_orfs, orf_finder=orf_finder)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        pbar = tqdm(
            total=len(files),
            desc="Predicting ORFs for each file",
            unit="file",
            file=LOGGER,
        )

        for file, genelist in zip(files, pool.map(find_orfs_fn, files)):
            protein_output = get_output_name(file, outdir, ".faa")
            genes_output = get_output_name(file, outdir, ".ffn")

            with ExitStack() as ctx:
                protein_fp = ctx.enter_context(protein_output.open("w"))
                genes_fp = (
                    ctx.enter_context(genes_output.open("w")) if write_genes else None
                )

                for scaffold_header, genes in zip(
                    FastaFile(file).parse_headers(), genelist
                ):

                    genes.write_translations(
                        protein_fp, sequence_id=scaffold_header.name, width=FASTA_WIDTH
                    )

                    if genes_fp is not None:
                        genes.write_genes(
                            genes_fp,
                            sequence_id=scaffold_header.name,
                            width=FASTA_WIDTH,
                        )

            pbar.update()
    pbar.close()


def main():
    args = parse_args()
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
        )
    else:
        # this is likely for MAGs so each file is a single genome composed of multiple scaffolds
        n_threads = min(len(files), max_cpus)
        find_orfs_multiple_files(
            files=files,
            orf_finder=orf_finder,
            outdir=outdir,
            write_genes=write_genes,
            n_threads=n_threads,
        )

    logging.info(f"Finished predicting ORFs for {len(files)} file(s).")


if __name__ == "__main__":
    main()
