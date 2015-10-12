# Splitting Compounds by Semantic Analogy

A compound splitter based on the semantic regularities in the vector space of word embeddings.
For more information see [this presentation](http://jodaiber.github.io/doc/compound_analogy_slides.pdf) or [our paper](http://jodaiber.github.io/doc/compound_analogy.pdf).


## Coming soon

- precomputed models
- fast and efficient splitter

# Usage

## Basic usage

To use this tool with standard settings, do the following:

```bash
$ wget https://raw.githubusercontent.com/jodaiber/semantic_compound_splitting/master/decompound_dict.py https://raw.githubusercontent.com/jodaiber/semantic_compound_splitting/master/models/de.dict
$ python decompound_dict.py de.dict < your_file > your_decompounded_file
```

The file `your_file` should contain tokenized sentences.


## Advanced usage


## Citation

If you use this splitter in your work, please cite:

```
@inproceedings{daiber2015compoundsplitting,
  title={Splitting Compounds by Semantic Analogy},
  author={Daiber, Joachim and Quiroz, Lautaro and Wechsler, Roger and Frank, Stella},
  booktitle={Proceedings of the 1st Deep Machine Translation Workshop},
  editor = {Jan Haji&#269; and António Branco},
  pages={20--28},
  year={2015},
  isbn = {978-80-904571-7-1},
  publisher={Charles University in Prague, Faculty of Mathematics and Physics, Institute of Formal and Applied Linguistics},
  url={http://jodaiber.github.io/doc/compound_analogy.pdf}
}
```


## Contributers

- Roger Wechsler, University of Amsterdam
- Lautaro Quiroz, University of Amsterdam
- [Joachim Daiber](http://jodaiber.de), ILLC, University of Amsterdam


## License

Apache 2.0
