from pathlib import Path


def get_output_name(file: Path, outdir: Path, genes: bool = False) -> Path:
    """Get the output results name based on the input file and output directory.

    Args:
        file (Path): input fasta file
        outdir (Path): output directory
        genes (bool, optional): whether writing to a genes fasta file or protein. Defaults to False.

    Returns:
        Path: output path
    """
    suffix = ".ffn" if genes else ".faa"

    return outdir.joinpath(file.with_suffix(suffix).name)
