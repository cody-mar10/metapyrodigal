# README

## Introduction

This library is a simple wrapper of [pyrodigal](https://github.com/althonos/pyrodigal), which is a cythonized implementation of [prodigal](https://github.com/hyattpd/Prodigal/) that is orders of magnitudes faster.

Pyrodigal is mostly written for single genomes or FASTA files, so this tool was created to batch process metagenomic-scale datasets. Metagenomic data usually consists of large number of genome files for MAGs. Additionally, viral metagenomic datasets tend to store all single-scaffold viruses in a single file, which tends to be much larger than a typical single-genome FASTA file.

This tool uses different load balancing strategies to parallelize pyrodigal over large amounts of files (MAGs) or FASTA files that have a large number of scaffolds (viruses).

## Installation

## Install versioned releases

```bash
pip install metapyrodigal
```

## Install from source

```bash
git clone https://github.com/cody-mar10/metapyrodigal.git
cd metapyrodigal
pip install .
```

## Usage

This tool will overwrite the `pyrodigal` binary, so you can use the metagenome-focused binary that I created.

The help page from `pyrodigal -h` looks like this:

```txt
usage: pyrodigal [-h] (-i FILE [FILE ...] | -d DIR) [-o DIR] [-c INT] [--genes] [--virus-mode] [-x STR]
                 [--allow-unordered]

Find ORFs from query genomes using pyrodigal v3.5.2, the cythonized prodigal API

options:
  -h, --help            show this help message and exit
  -i FILE [FILE ...], --input FILE [FILE ...]
                        fasta file(s) of query genomes (can use unix wildcards)
  -d DIR, --input-dir DIR
                        directory of fasta files to process
  -o DIR, --outdir DIR  output directory (default: /storage2/scratch/ccmartin6/software/metapyrodigal)
  -c INT, --max-cpus INT
                        maximum number of threads to use (default: 1)
  --genes               use to also output the nucleotide genes .ffn file
  --virus-mode          use pyrodigal-gv to activate the virus models (default: False)
  -x STR, --extension STR
                        genome FASTA file extension if using -d/--input-dir (default: fna)
  --allow-unordered     for a single file input, this allows the protein ORFs to be written per scaffold as
                        available. All protein ORFs for each scaffold will be in order, but the scaffolds will not
                        necessarily be in the same order as in the input nucleotide file. **This is useful if you
                        are extremely memory limited,** since the default strategy can lead to the ORFs being
                        stored in memory for awhile before writing to file as the original scaffold order is
                        maintained. NOTE: This is about 20 percent faster, so it is recommended to use this if the
                        order of scaffolds does not matter.
```

`-i` and `-d` are mutually exclusive but one of them must be provided.

The output files have the same basename as the input file. Protein FASTA files will have the extension `.faa`, and nucleotide gene FASTA files will have the extension `.ffn`. For example:

```bash
pyrodigal -i GENOME.fna
```

will output `GENOME.faa`
