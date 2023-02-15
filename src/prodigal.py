#!/usr/bin/env python3
import argparse
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from glob import iglob
from itertools import repeat
from pathlib import Path

import pyrodigal
from pyrodigal._version import __version__ as pyrodigal_version
from fastatools.fastaparser import fastaparser

MAX_PYRODIGAL_THREADS = 4
FASTA_WIDTH = 75


def _find_orfs(
    orf_finder: pyrodigal.OrfFinder, name: str, seq: str
) -> tuple[str, pyrodigal.Genes]:
    orfs = orf_finder.find_genes(seq)
    return name, orfs


def find_orfs(
    orf_finder: pyrodigal.OrfFinder,
    file: Path,
    outdir: Path,
    write_genes: bool,
    threads: int,
):
    names: list[str] = list()
    sequences: list[str] = list()
    for name, seq in fastaparser(file):
        names.append(name)
        sequences.append(seq)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        orfs = dict(executor.map(_find_orfs, repeat(orf_finder), names, sequences))

    # TODO: optionally write .ffn genes file
    output = outdir.joinpath(file.with_suffix(".faa").name)
    genes_output = outdir.joinpath(file.with_suffix(".ffn").name)

    if write_genes:
        log_msg = f"Writing nucleotide and protein ORFs for {file} to {outdir}"
    else:
        log_msg = f"Writing protein ORFs for {file} to {output}"

    logging.info(log_msg)

    with ExitStack() as ctx:
        ptn_fp = ctx.enter_context(output.open("w"))
        genes_fp = ctx.enter_context(genes_output.open("w")) if write_genes else None
        for name, genes in orfs.items():
            prefix = f"{name}_"
            genes.write_translations(ptn_fp, prefix=prefix, width=FASTA_WIDTH)
            if genes_fp is not None:
                genes.write_genes(genes_fp, prefix=prefix, width=FASTA_WIDTH)


def main(
    files: list[Path],
    outdir: Path,
    write_genes: bool,
    threads: int,
    max_cpus: int,
    log: Path,
):
    if len(files) == 1:
        # TODO: should make this overrideable
        outdir = Path.cwd()
    outdir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        filename=log,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(
        f"Predicting ORFs for {len(files)} files using Pyrodigal v{pyrodigal_version}"
    )
    orf_finder = pyrodigal.OrfFinder(meta=True, mask=True)

    if len(files) == 1:
        # if one file, use max pyrodigal threads for best performance
        threads = MAX_PYRODIGAL_THREADS
    else:
        threads = min(threads, MAX_PYRODIGAL_THREADS)

    n_proc = max_cpus // threads
    if n_proc == 0:
        # basically if max_cpus < threads
        n_proc = 1
    else:
        n_proc = min(len(files), n_proc)

    with multiprocessing.Pool(processes=n_proc) as pool:
        pool.starmap(
            find_orfs,
            zip(
                repeat(orf_finder),
                files,
                repeat(outdir),
                repeat(write_genes),
                repeat(threads),
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find ORFs from query genomes using pyrodigal, the cythonized prodigal API"
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="fasta file(s) of query genomes (can use unix wildcards)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=Path.cwd(),
        type=Path,
        help="output directory - If you only are predicting for a single file, this will automatically become the cwd (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="number of parallel threads for orf prediction. Max value is 4, with best results for multiple files being 1 thread per process. You should not usually change this, although for single file inputs, this will always be changed to 4 for best performance. See -c/--cpu option, which specifies the maximum CPU usage. (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--max-cpus",
        type=int,
        default=20,
        help="number of files to process in parallel. This controls the MAXIMUM CPU usage. It is best to make this a multiple of -t/--threads; otherwise, your max usage will be rounded down. (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default="pyrodigal.log",
        help="log file (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        action="store_true",
        help="use if -i/--input is a glob pattern and the shell could not expand (ie if too many files) (default: %(default)s)",
    )
    parser.add_argument(
        "--genes",
        action="store_true",
        help="use to also output the nucleotide genes .ffn file (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.pattern:
        files = [Path(file) for file in iglob(args.input[0].as_posix())]
    else:
        files: list[Path] = args.input

    main(
        files=files,
        outdir=args.outdir,
        write_genes=args.genes,
        threads=args.threads,
        max_cpus=args.max_cpus,
        log=args.log,
    )
