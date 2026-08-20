from capymoa.datasets._downloader import _DownloadableARFF


class Sensor(_DownloadableARFF):
    """Sensor stream is a classification problem based on indoor sensor data.

    * Number of instances: 2,219,803
    * Number of attributes: 5
    * Number of classes: 54

    The stream contains temperature, humidity, light, and sensor voltage
    collected from 54 sensors deployed in Intel Berkeley Research Lab. The
    classification objective is to predict the sensor ID.

    **References:**

    #.  https://www.cse.fau.edu/~xqzhu/stream.html
    """

    _target_type = "categorical"
    _length = 2219803


class Hyper100k(_DownloadableARFF):
    """Hyper100k is a classification problem based on the moving hyperplane generator.

    * Number of instances: 100,000
    * Number of attributes: 10
    * Number of classes: 2

    **References:**

    #.  Hulten, Geoff, Laurie Spencer, and Pedro Domingos. "Mining time-changing
        data streams." Proceedings of the seventh ACM SIGKDD international conference
        son Knowledge discovery and data mining. 2001.
    """

    # TODO: Add docstring describing the dataset and link to the original source

    _target_type = "categorical"
    _length = 100_000


class CovtFD(_DownloadableARFF):
    """CovtFD is an adaptation from the classic :class:`Covtype` classification
    problem with added feature drifts.

    * Number of instances: 581,011 (30m^2 cells)
    * Number of attributes: 104 (10 continuous, 44 categorical, 50 dummy)
    * Number of classes: 7 (forest cover types)

    Given 30x30-meter cells obtained from the US Resource Information System
    (RIS). The dataset includes 10 continuous and 44 categorical features, which
    we augmented by adding 50 dummy continuous features drawn from a Normal
    probability distribution with μ = 0 and σ = 1. Only the continuous features
    were randomly swapped with 10 (out of the fifty) dummy features to simulate
    drifts. We added such synthetic drift twice, one at instance 193, 669 and
    another at 387, 338.

    **References:**

    #.  Gomes, Heitor Murilo, Rodrigo Fernandes de Mello, Bernhard Pfahringer,
        and Albert Bifet. "Feature scoring using tree-based ensembles for
        evolving data streams." In 2019 IEEE International Conference on
        Big Data (Big Data), pp. 761-769. IEEE, 2019.
    #.  Blackard,Jock. (1998). Covertype. UCI Machine Learning Repository.
        https://doi.org/10.24432/C50K5N.
    #.  https://archive.ics.uci.edu/ml/datasets/Covertype

    **See Also:**

    * :class:`Covtype` - The classic covertype dataset
    * :class:`CovtypeNorm` - A normalized version of the classic covertype dataset
    * :class:`CovtypeTiny` - A truncated version of the classic covertype dataset
    """

    _target_type = "categorical"
    _length = 581_011


class Covtype(_DownloadableARFF):
    """The classic covertype (/covtype) classification problem

    * Number of instances: 581,012 (30m^2 cells)
    * Number of attributes: 54 (10 continuous, 44 categorical)
    * Number of classes: 7 (forest cover types)

    Forest Covertype (or simply covtype) contains the forest cover type for 30 x 30
    meter cells obtained from US Forest Service (USFS) Region 2 Resource
    Information System (RIS) data.

    **References:**

    #.  Blackard,Jock. (1998). Covertype. UCI Machine Learning Repository.
        https://doi.org/10.24432/C50K5N.
    #.  https://archive.ics.uci.edu/ml/datasets/Covertype

    **See Also:**

    * :class:`CovtFD` - Covtype with simulated feature drifts
    * :class:`CovtypeNorm` - A normalized version of the classic covertype dataset
    * :class:`CovtypeTiny` - A truncated version of the classic covertype dataset
    """

    _target_type = "categorical"
    _length = 581_012


class CovtypeTiny(_DownloadableARFF):
    """A truncated version of the classic :class:`Covtype` classification problem.

    **This should only be used for quick tests, not for benchmarking algorithms.**

    * Number of instances: first 1001 (30m^2 cells)
    * Number of attributes: 54 (10 continuous, 44 categorical)
    * Number of classes: 7 (forest cover types)

    Forest Covertype (or simply covtype) contains the forest cover type for 30 x 30
    meter cells obtained from US Forest Service (USFS) Region 2 Resource
    Information System (RIS) data.

    **References:**

    #.  Blackard,Jock. (1998). Covertype. UCI Machine Learning Repository.
        https://doi.org/10.24432/C50K5N.
    #.  https://archive.ics.uci.edu/ml/datasets/Covertype

    **See Also:**

    * :class:`CovtFD` - Covtype with simulated feature drifts
    * :class:`Covtype` - The classic covertype dataset
    * :class:`CovtypeNorm` - A normalized version of the classic covertype dataset
    """

    _target_type = "categorical"
    _length = 1001


class CovtypeNorm(_DownloadableARFF):
    """A normalized version of the classic :class:`Covtype` classification problem.

    * Number of instances: 581,012 (30m^2 cells)
    * Number of attributes: 54 (10 continuous, 44 categorical)
    * Number of classes: 7 (forest cover types)

    Forest Covertype (or simply covtype) contains the forest cover type for 30 x 30
    meter cells obtained from US Forest Service (USFS) Region 2 Resource
    Information System (RIS) data.

    **References:**

    #.  Blackard,Jock. (1998). Covertype. UCI Machine Learning Repository.
        https://doi.org/10.24432/C50K5N.
    #.  https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/covtypeNorm.arff.zip/download/


    **See Also:**

    * :class:`CovtFD` - Covtype with simulated feature drifts
    * :class:`Covtype` - The classic covertype dataset
    * :class:`CovtypeTiny` - A truncated version of the classic covertype dataset
    """

    _target_type = "categorical"
    _length = 581_012


class RBFm_100k(_DownloadableARFF):
    """RBFm_100k is a synthetic classification problem based on the Radial
    Basis Function generator.

    * Number of instances: 100,000
    * Number of attributes: 10
    * ``generators.RandomRBFGeneratorDrift -s 1.0E-4 -c 5``

    This is a snapshot (100k instances) of the synthetic generator RBF
    (Radial Basis Function), which works as follows: A fixed number of random
    centroids are generated. Each center has a random position, a single
    standard deviation, class label and weight. New examples are generated by
    selecting a center at random, taking weights into consideration so that
    centers with higher weight are more likely to be chosen. A random direction
    is chosen to offset the attribute values from the central point. The length
    of the displacement is randomly drawn from a Gaussian distribution with
    standard deviation determined by the chosen centroid. The chosen centroid
    also determines the class label of the example. This effectively creates a
    normally distributed hypersphere of examples surrounding each central point
    with varying densities. Only numeric attributes are generated.
    """

    _target_type = "categorical"
    _length = 100_000


class RTG_2abrupt(_DownloadableARFF):
    """RTG_2abrupt is a synthetic classification problem based on the Random Tree
    generator with 2 abrupt drifts.

    * Number of instances: 100,000
    * Number of attributes: 30
    * Number of classes: 5
    * ``generators.RandomTreeGenerator -o 0 -u 30 -d 20``

    This is a snapshot (100k instances with 2 simulated abrupt drifts) of the
    synthetic generator based on the one proposed by Domingos and Hulten [1],
    producing concepts that in theory should favour decision tree learners.
    It constructs a decision tree by choosing attributes at random to split,
    and assigning a random class label to each leaf. Once the tree is built,
    new examples are generated by assigning uniformly distributed random values
    to attributes which then determine the class label via the tree.

    **References:**

    #.  Domingos, Pedro, and Geoff Hulten. "Mining high-speed data streams."
        In Proceedings of the sixth ACM SIGKDD international conference on
        Knowledge discovery and data mining, pp. 71-80. 2000.

    See also :class:`capymoa.stream.generator.RandomTreeGenerator`
    """

    _target_type = "categorical"
    _length = 100_000


class Electricity(_DownloadableARFF):
    """Electricity is a classification problem based on the Australian New
    South Wales Electricity Market.

    * Number of instances: 45,312
    * Number of attributes: 8
    * Number of classes: 2 (UP, DOWN)

    The Electricity data set was collected from the Australian New South Wales
    Electricity Market, where prices are not fixed. It was described by M.
    Harries and analysed by Gama. These prices are affected by demand and supply
    of the market itself and set every five minutes. The Electricity data set
    contains 45,312 instances, where class labels identify the changes of the
    price (2 possible classes: up or down) relative to a moving average of the
    last 24 hours. An important aspect of this data set is that it exhibits
    temporal dependencies. This version of the dataset has been normalised (AKA
    ``elecNormNew``) and it is the one most commonly used in benchmarks.

    **References:**

    #.  https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/elecNormNew.arff.zip/download/

    """

    _target_type = "categorical"
    _length = 45_312


class ElectricityTiny(_DownloadableARFF):
    """A truncated version of the Electricity dataset with 1000 instances.

    This is a tiny version (2k instances) of the Electricity widely used dataset
    described by M. Harries. **This should only be used for quick tests, not for
    benchmarking algorithms.**

    See :class:`Electricity` for the widely used electricity dataset.
    """

    _target_type = "categorical"
    _length = 2_000


class Fried(_DownloadableARFF):
    """Fried is a regression problem based on the Friedman dataset.

    * Number of instances: 40,768
    * Number of attributes: 10
    * Number of targets: 1

    This is an artificial dataset that contains ten features, only five out of
    which are related to the target value.

    **References:**

    #.  Friedman, Jerome H. "Multivariate adaptive regression splines." The
        annals of statistics 19, no. 1 (1991): 1-67.
    """

    _target_type = "numeric"
    _length = 40_768


class FriedTiny(_DownloadableARFF):
    """A truncated version of the Friedman regression problem with 1000 instances.

    This is a tiny version (1k instances) of the Fried dataset. **This should
    only be used for quick tests, not for benchmarking algorithms.**

    See :class:`Fried` for the full Friedman dataset.
    """

    _target_type = "numeric"
    _length = 1_000


class Bike(_DownloadableARFF):
    """Bike is a regression dataset for the amount of bike share information.

    * Number of instances: 17,379
    * Number of attributes: 12
    * Number of targets: 1

    This dataset contains the hourly and daily count of rental bikes
    between years 2011 and 2012 in Capital bike share system with the
    corresponding weather and seasonal information.

    **References:**
    #.  Fanaee-T, Hadi, and Joao Gama. "Event labeling combining ensemble detectors
    and background knowledge." Progress in Artificial Intelligence 2 (2014): 113-127.
    """

    _target_type = "numeric"
    _length = 17_379


class Airlines(_DownloadableARFF):
    """Airlines dataset inspired in the regression dataset from Elena Ikonomovska.

    * Number of instances: 539,383
    * Number of attributes: 8
    * Number of targets: 2

    The task is to predict whether a given flight will be delayed, given the information
    of the scheduled departure.

    **References:**

    #.  Ikonomovska, Elena. "Airline Data Set." Data Expo Competition (2009):
        http://kt.ijs.si/elena_ikonomovska/data.html (archived:
        https://web.archive.org/web/20110718072348/http://kt.ijs.si/elena_ikonomovska/data.html).
    #.  Bifet, Albert, and Elena Ikonomovska. "airlines." OpenML (2014):
        https://www.openml.org/d/1169.
    """

    _target_type = "categorical"
    _length = 539383


class KDD99(_DownloadableARFF):
    """KDD99 is a network intrusion detection problem based on a 10% stratified
    subsample of the 1998 DARPA Intrusion Detection Evaluation Program data.

    * Number of instances: 494,020
    * Number of attributes: 41
    * Number of classes: 23

    The task is to distinguish between normal connections and different types of
    network intrusions (attacks), grouped into four main categories: denial-of-service,
    unauthorized access from a remote machine, unauthorized access to local
    superuser privileges, and surveillance/probing.

    **References:**

    #.  Stolfo, Salvatore, Wei Fan, Wenke Lee, Andreas Prodromidis, and Philip
        Chan. "KDD Cup 1999 Data." UCI Machine Learning Repository (1999):
        https://doi.org/10.24432/C51C7N.
    #.  "KDDCup99." OpenML (2014): https://www.openml.org/d/1113.
    """

    _target_type = "categorical"
    _length = 494_020


class Nomao(_DownloadableARFF):
    """Nomao is a deduplication classification problem based on data aggregated
    by the Nomao Labs search engine.

    * Number of instances: 34,465
    * Number of attributes: 118
    * Number of classes: 2

    The task is to determine whether two place records (containing information
    such as names, phone numbers, and addresses collected from several sources)
    refer to the same real-world place. The attributes measure similarity and
    matching characteristics across the various fields of the records being compared.

    **References:**

    #.  Candillier, Laurent, and Vincent Lemaire. "Design and analysis of the
        Nomao challenge active learning in the real-world." Proceedings of the
        ALRA: Active Learning in Real-world Applications, Workshop ECML-PKDD
        (2012): https://archive.ics.uci.edu/ml/datasets/Nomao.
    #.  "nomao." OpenML (2015): https://www.openml.org/d/1486.
    """

    _target_type = "categorical"
    _length = 34_465


class Spambase(_DownloadableARFF):
    """Spambase is a classification problem based on the classic UCI Spambase
    email dataset from Hewlett-Packard Labs.

    * Number of instances: 4,601
    * Number of attributes: 57
    * Number of classes: 2

    The task is to predict whether a given email is spam. Spam emails were
    collected from postmaster and individual submissions, while non-spam
    emails came from filed work and personal correspondence. Attributes
    consist of word and character frequency measures as well as measures of
    the length of consecutive sequences of capital letters.

    **References:**

    #.  Hopkins, Mark, Erik Reeber, George Forman, and Jaap Suermondt.
        "Spambase." UCI Machine Learning Repository (1999):
        https://archive.ics.uci.edu/dataset/94/spambase,
        https://doi.org/10.24432/C53G6X.
    #.  "spambase." OpenML (2014): https://www.openml.org/d/44.
    """

    _target_type = "categorical"
    _length = 4_601


class PokerHand(_DownloadableARFF):
    """PokerHand is a classification problem where each instance is an example
    of a hand consisting of five playing cards drawn from a standard deck of 52.

    * Number of instances: 1,025,009
    * Number of attributes: 10
    * Number of classes: 10

    Each card is described using two attributes (suit and rank), for a total
    of 10 predictive attributes. The task is to predict the poker hand,
    ranging from nothing to royal flush. Note that the order of cards is
    important, so there are 480 possible Royal Flush hands instead of just 4.

    **References:**

    #.  Cattral, Robert, Franz Oppacher, and Dwight Deugo. "Evolutionary data
        mining with automatic rule generalization." Recent Advances in
        Computers, Computing and Communications (2002).
    #.  "poker-hand." OpenML (2015): https://www.openml.org/d/1567.
    """

    _target_type = "categorical"
    _length = 1_025_009
