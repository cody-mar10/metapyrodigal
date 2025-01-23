import argparse
import logging
import sys
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional

from pyrodigal._version import __version__ as pyrodigal_version

from metapyrodigal.load_balancer import SingleFileLoadBalancer, load_balancer
from metapyrodigal.orf_finder import OrfFinder

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
        fields = {field.name: getattr(namespace, field.name) for field in dataclass_fields(cls)}

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
    msg = f"Predicting ORFs for {len(files)} file(s) using Pyrodigal v{pyrodigal_version}"

    if args.virus_mode:
        msg += " with virus mode enabled (ie pyrodigal-gv)"

    logging.info(msg)
    logging.info(f"Using up to {max_cpus} thread(s)")
    logging.info("Using C++ implemented fastaparser from pyfastatools")

    orf_finder = OrfFinder(virus_mode=args.virus_mode)

    with load_balancer(
        files, orf_finder=orf_finder, allow_unordered=args.allow_unordered, n_threads=max_cpus
    ) as balancer:
        if isinstance(balancer, SingleFileLoadBalancer):
            msg = "Using single file load balancer with {ordering} processing"
            msg = msg.format(ordering="unordered" if args.allow_unordered else "ordered")
        else:
            msg = "Using multi file load balancer"

        logging.info(msg)

        balancer.submit_to_pool(files, outdir, write_genes)

    # logging.info("Using C++ implemented fastaparser")

    logging.info(f"Finished predicting ORFs for {len(files)} file(s).")


if __name__ == "__main__":
    main()
