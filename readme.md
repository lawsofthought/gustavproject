Gustav: A probabilistic topic modelling toolbox
===============================================

Author

:   Mark Andrews

Disclaimer

:   This is alpha software.

Gustav is a Python and Fortran based probabilistic topic modelling
toolbox.

It is released under [GNU General Public
License](http://www.gnu.org/copyleft/gpl.html).

The name `Gustav` is named after [Peter Gustav Lejeune
Dirichlet](https://en.wikipedia.org/wiki/Peter_Gustav_Lejeune_Dirichlet),
the 19th century German mathematician after whom the [Dirichlet
Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) and
[Dirichlet Process](https://en.wikipedia.org/wiki/Dirichlet_process) are
named. Both the [Dirichlet
Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) and
[Dirichlet Process](https://en.wikipedia.org/wiki/Dirichlet_process) are
vital components of probabilistic topic models.

# Installation

It is recommended that Gustav is installed using
[pip](https://pip.pypa.io) (and in a [virtual
environment](https://virtualenv.pypa.io), though that is a matter of
preference).

``` {.bash}
make all                # compile fortran extension modules
python setup.py test    # optional
pip install -e . 
```

# Warning: Alpha software

Currently, Gustav is a alpha software.

-   It implements only a minimal number of probabilistic topic models so
    far.
-   The API to any given sampler or model may change without warning.
-   Any future development is likely to be backwards incompatible.
-   There is minimial documenation.
-   In short, use with caution.



# Modelling with Gustav

The following is the usage notes for `gustav`: 

```bash
Gustav: Probabilistic Topic Modelling Toolbox

Usage:
  gustave model new [--model-type=<model_type>] <corpus_name> [--K_min=<K_min>] [--K_max=<K_max>]
  gustave model <model_name> update [--iterations=<N>] [--hyperparameters] [--parallel=<K>]
  gustave data new <corpus_name> [--data-type=<data_type>] <text_file> <vocab_file>
  gustave init 
  gustave (-h | --help)
  gustave --version

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --parallel=<K>                Number of processors [default: 1]
  --iterations=<N>              Model update iterations [default: 100]
  --model-type=<model_type>     Type of topic model [default: hdptm].
  --data-type=<data_type>       Type of data set [default: bag_of_words].
  --K_min=<K_min>               Minimum number of topics [default: 10]
  --K_max=<K_max>               Maximum number of topics [default: 100]
```
## Example: Create a corpus

```bash
gustave data new foo example_corpus.txt vocab.txt
```
where `example_corpus.txt` is a text corpus where the "texts" are delimited by
line breaks and the "words" are delimited by "|", e.g. 
```bash
foo|bar|foobar|foo|foo
foobar|foo|bar|bar|bar
bar|foo|bar|foo|bar
```
and `vocab.txt` is a line break delimited list of word types, e.g. 
```bash
foo
bar
foobar
```

## Example: Initialize your topic model using corpus `foo`

```bash
gustave model new foo --K_min=1000 --K_max=2500
```

The created model will be given a random name like `hdptm_180117202450_6333` where the first string of digits is datetimestamp and the second is random integer.

## Example: Update the topic model

Update the model for 1000 iterations.
```bash
gustave model hdptm_180117202450_6333 update --parallel 16 --iterations=1000 --hyperparameters
```

## Saving results

Corpora and samples are saved inside a directory called `data`, and all details are stored in the config file `gustav.cfg`.
