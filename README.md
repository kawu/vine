Vine
=====

Vine is a neural graph-based [verbal MWE identification][parseme-st] tool built
on top of the [backprop][backprop] library.


Installation
------------

First you will need to download and install the [Haskell Tool Stack][stack].
Then run the following commands:

    git clone https://github.com/kawu/vine.git
    cd vine
    stack install

The above command builds the `vine` command-line tool and (on Linux) puts it in
the `~/.local/bin/` directory by default.


Data format
-----------

Vine works with the PARSEME [.cupt][cupt] (extended CoNLL-U) format.


Prerequisites
-------------

Before you can use the tool to identify verbal MWEs (VMWEs) in a given
language, you will need to train the appropriate identification models.
To this end, you will need:

  * A training file with VMWE annotations in the [.cupt][cupt] format
  * A file with the word embedding vectors correspondig to the individual words
    in the training file

You can find an example training set in the `example` directory.

### Configuration

You need to specify the training configuration to train a model.  The
configuration includes the SGD parameters and the type of the objective
function to optimize.  An example configuration can be found in the
`example/config.dhall` file.  The configuration is written in [Dhall][dhall].

*WARNING*: the example `config.dhall` file will work well with the example
training files.  If you want to train a model on a larger dataset, consider
using the `example/config.real.dhall` configuration file with a smaller
stepsize and a larger mini-batch size.


Training
--------

Vine requires training one model per VMWE category (`VID` -- verbal idiom,
`LVC.full` -- light-verb construction, etc.).  See the [PARSEME annotation
guidelines][PARSEME-annotation-guidelines] for a description of the different
categories employed in the PARSEME annotation framework.

Given:

  * Training file `train.cupt` in the [.cupt][cupt] format
  * File with the corresponding training word emeddings `train.embs`
  * Training configuration `config.dhall`

You can use the following command to train a model for the identification of
light verbs:

    vine train -i train.cupt -e train.embs --mwe LVC.full --sgd config.dhall -o models/LVC.full.bin

where the `-o` option specifies the output model file.  The tool will also save
the intermediate model files (`LVC.full.bin.N`, where `N` is the number of the
iteration) in the target `models` directory.  You have to create this directory
manually before training.

While training, the tool reports the value of the objective function, which
should reach a relatively low level at the end of the process.  In case of the
example training set, the final objective value should be very close to 0,
which represents a perfect fit of the model to the training data (modulo VMWE
encoding errors).


### Runtime options

You can use the [GHC runtime system options][ghc-rts] to speed up training.
For instance, you can make use of multiple cores using `-N`, or increase the
allocation area size using `-A`. For example, to train the model using four
threads and 256M allocation area size, run:

    vine train -i train.cupt -e train.embs --mwe LVC.full --sgd config.dhall -o models/LVC.full.bin +RTS -A256M -N4


Tagging
-------

Use the following command to tag a given `test.blind.cupt` file (`blind` is
used here to mark a file with no VMWE annotations) with light verbs using a
`LVC.full.bin` model file:

    vine tag --constrained -t LVC.full -m LVC.full.bin -i test.blind.cupt -e test.embs

where `test.embs` is a file with the embedding vectors corresponding to the
words in `test.blind.cupt`, and the `--constrained` (`-c`) option specifies
that constrained prediction is to be used.

### Clearing

The `tag` command does not erase the VMWE annotations present in the input
file.  If you want to tag a file already containing VMWE annotations, you can
`clear` it first:

    vine clear < test.cupt > test.blind.cupt

### Ensemble averaging

The quality of the identification models can vary between the different
training runs.  To counteract this issue, you can train several models and use
them together for tagging, for instance:

    vine tag -c -t LVC.full -m LVC.full.1.bin -m LVC.full.2.bin -m LVC.full.3.bin -i test.blind.cupt -e test.embs


PARSEME-ST
----------

Coming soon.


[stack]: http://docs.haskellstack.org "Haskell Tool Stack"
[backprop]: https://backprop.jle.im/index.html "Backpropagation library"
[parseme-st]: http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task "PARSEME Shared Task"
[cupt]: http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_45_Format_specification "PARSEME .cupt format"
[ghc-rts]: http://www.haskell.org/ghc/docs/latest/html/users_guide/runtime_control.html "GHC runtime system options"
[dhall]: https://github.com/dhall-lang/dhall-lang "Dhall"
[PARSEME-annotation-guidelines]: http://parsemefr.lif.univ-mrs.fr/parseme-st-guidelines/1.1/ "PARSEME annotation guidelines"
