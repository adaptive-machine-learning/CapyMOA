from capymoa.datasets.downloader import DownloadARFFGzip
from ._source_list import SOURCE_LIST
from ._utils import identify_compressed_hosted_file


def _get_filename(source_name: str):
    return identify_compressed_hosted_file(SOURCE_LIST[source_name].arff)[1]


def _get_remote_url(source_name: str):
    return SOURCE_LIST[source_name].arff


class Sensor(DownloadARFFGzip):
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

    _filename = _get_filename("Sensor")
    _remote_url = _get_remote_url("Sensor")
    _length = 2219803


class Hyper100k(DownloadARFFGzip):
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

    _filename = _get_filename("Hyper100k")
    _remote_url = _get_remote_url("Hyper100k")
    _length = 100_000


class CovtFD(DownloadARFFGzip):
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

    _filename = _get_filename("CovtFD")
    _remote_url = _get_remote_url("CovtFD")
    _length = 581_011


class Covtype(DownloadARFFGzip):
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

    _filename = _get_filename("Covtype")
    _remote_url = _get_remote_url("Covtype")
    _length = 581_012


class CovtypeTiny(DownloadARFFGzip):
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

    _filename = _get_filename("CovtypeTiny")
    _remote_url = _get_remote_url("CovtypeTiny")
    _length = 1001


class CovtypeNorm(DownloadARFFGzip):
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

    _filename = _get_filename("CovtypeNorm")
    _remote_url = _get_remote_url("CovtypeNorm")
    _length = 581_012


class RBFm_100k(DownloadARFFGzip):
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

    _filename = _get_filename("RBFm_100k")
    _remote_url = _get_remote_url("RBFm_100k")
    _length = 100_000


class RTG_2abrupt(DownloadARFFGzip):
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

    _filename = _get_filename("RTG_2abrupt")
    _remote_url = _get_remote_url("RTG_2abrupt")
    _length = 100_000


class Electricity(DownloadARFFGzip):
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

    _filename = _get_filename("Electricity")
    _remote_url = _get_remote_url("Electricity")
    _length = 45_312


class ElectricityTiny(DownloadARFFGzip):
    """A truncated version of the Electricity dataset with 1000 instances.

    This is a tiny version (2k instances) of the Electricity widely used dataset
    described by M. Harries. **This should only be used for quick tests, not for
    benchmarking algorithms.**

    See :class:`Electricity` for the widely used electricity dataset.
    """

    _filename = _get_filename("ElectricityTiny")
    _remote_url = _get_remote_url("ElectricityTiny")
    _length = 2_000


class Fried(DownloadARFFGzip):
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

    _filename = _get_filename("Fried")
    _remote_url = _get_remote_url("Fried")
    _length = 40_768


class FriedTiny(DownloadARFFGzip):
    """A truncated version of the Friedman regression problem with 1000 instances.

    This is a tiny version (1k instances) of the Fried dataset. **This should
    only be used for quick tests, not for benchmarking algorithms.**

    See :class:`Fried` for the full Friedman dataset.
    """

    _filename = _get_filename("FriedTiny")
    _remote_url = _get_remote_url("FriedTiny")
    _length = 1_000


class Bike(DownloadARFFGzip):
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

    _filename = _get_filename("Bike")
    _remote_url = _get_remote_url("Bike")
    _length = 17_379
