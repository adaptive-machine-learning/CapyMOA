# SPDX-License-Identifier: BSD-3-Clause
from openmoa.datasets.downloader import DownloadARFFGzip
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

    See also :class:`openmoa.stream.generator.RandomTreeGenerator`
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


# ==========================================
# New Benchmark Datasets (Binary & Multi-Class)
# ==========================================

class RCV1(DownloadARFFGzip):
    """RCV1 (Reuters Corpus Volume I) is a text classification dataset.
    The binary version typically used in benchmarks distinguishes between
    CCAT (Corporate/Industrial) and ECAT (Economics) categories.

    * Number of instances: 20,242 (standard binary version)
    * Number of attributes: ~47,236 (sparse)
    * Number of classes: 2

    **References:**

    #.  Lewis, David D., et al. "RCV1: A new benchmark collection for text
        categorization research." Journal of machine learning research 5.Apr (2004): 361-397.
    """

    _filename = _get_filename("RCV1")
    _remote_url = _get_remote_url("RCV1")
    _length = 20_242


class W8a(DownloadARFFGzip):
    """w8a is a sparse binary classification dataset derived from web data.
    It is a larger version of the w1a-w7a series.

    * Number of instances: ~49,749
    * Number of attributes: 300
    * Number of classes: 2

    **References:**

    #.  Platt, John C. "Fast training of support vector machines using sequential minimal optimization."
        Advances in kernel methods. MIT press, 1999.
    """

    _filename = _get_filename("W8a")
    _remote_url = _get_remote_url("W8a")
    _length = 49_749


class Adult(DownloadARFFGzip):
    """Adult (also known as a8a in LIBSVM) is a classification dataset to predict
    whether income exceeds $50K/yr based on census data.

    * Number of instances: 32,561
    * Number of attributes: 123 (one-hot encoded) / 14 (original)
    * Number of classes: 2

    **References:**

    #.  Kohavi, Ron. "Scaling up the accuracy of naive-bayes classifiers: A
        decision-tree hybrid." KDD. Vol. 96. 1996.
    #.  https://archive.ics.uci.edu/ml/datasets/Adult
    """

    _filename = _get_filename("Adult")
    _remote_url = _get_remote_url("Adult")
    _length = 32_561


class Magic04(DownloadARFFGzip):
    """MAGIC Gamma Telescope dataset. The task is to classify images into
    high energy gamma particles (g) or background hadron (h).

    * Number of instances: 19,020
    * Number of attributes: 10 (real)
    * Number of classes: 2

    **References:**

    #.  Bock, R. K., et al. "Methods for multidimensional event classification:
        a case study using images from a Cherenkov gamma-ray telescope."
        Nuclear Instruments and Methods in Physics Research Section A. 2004.
    """

    _filename = _get_filename("Magic04")
    _remote_url = _get_remote_url("Magic04")
    _length = 19_020


class Spambase(DownloadARFFGzip):
    """Spambase is a classic email spam classification dataset.

    * Number of instances: 4,601
    * Number of attributes: 57 (continuous)
    * Number of classes: 2 (spam vs non-spam)

    The attributes measure the frequency of certain words and characters in emails.

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Spambase
    """

    _filename = _get_filename("Spambase")
    _remote_url = _get_remote_url("Spambase")
    _length = 4_601


class Musk(DownloadARFFGzip):
    """Musk (Version 2) dataset. The task is to predict whether a new molecule
    will be a musk or a non-musk.

    * Number of instances: ~6,598 (full) or ~3,062 (standard subset)
    * Number of attributes: 166
    * Number of classes: 2

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)
    """

    _filename = _get_filename("Musk")
    _remote_url = _get_remote_url("Musk")
    # Using the standard subset size commonly found in benchmarks
    _length = 6_598

class SVMGuide3(DownloadARFFGzip):
    """SVMGuide3 is a dataset from the LIBSVM guide, often used to
    demonstrate scaling and kernel performance.

    * Number of instances: ~1,243
    * Number of attributes: 21
    * Number of classes: 2

    **References:**

    #.  Hsu, Chih-Wei, Chih-Jen Lin, and Bor-Chen Juand. "A practical guide to
        support vector classification." (2003).
    """

    _filename = _get_filename("SVMGuide3")
    _remote_url = _get_remote_url("SVMGuide3")
    _length = 1_243


class German(DownloadARFFGzip):
    """German Credit Data. Classifies people as good or bad credit risks.

    * Number of instances: 1,000
    * Number of attributes: 24 (numerical representation)
    * Number of classes: 2

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    """

    _filename = _get_filename("German")
    _remote_url = _get_remote_url("German")
    _length = 1_000


class Australian(DownloadARFFGzip):
    """Australian Credit Approval dataset.

    * Number of instances: 690
    * Number of attributes: 14
    * Number of classes: 2

    All attribute names and values have been changed to meaningless symbols
    to protect confidentiality.

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Statlog+(Australian+Credit+Approval)
    """

    _filename = _get_filename("Australian")
    _remote_url = _get_remote_url("Australian")
    _length = 690


class Ionosphere(DownloadARFFGzip):
    """Ionosphere dataset. Classification of radar returns from the ionosphere.
    Good radar returns are those showing evidence of some type of structure in
    the ionosphere.

    * Number of instances: 351
    * Number of attributes: 34
    * Number of classes: 2 (Good/Bad)

    **References:**

    #.  Sigillito, V. G., et al. "Classification of radar returns from the ionosphere
        using neural networks." Johns Hopkins APL Technical Digest 10 (1989): 262-266.
    """

    _filename = _get_filename("Ionosphere")
    _remote_url = _get_remote_url("Ionosphere")
    _length = 351


class InternetAds(DownloadARFFGzip):
    """Internet Advertisements dataset. The task is to predict whether an
    image is an advertisement ("ad") or not ("nonad").

    * Number of instances: 3,279 (before cleaning) / ~1,960 (standard subset)
    * Number of attributes: 1,558
    * Number of classes: 2 (or 6 in specific multi-class versions)

    **References:**

    #.  Kushmerick, Nicholas. "Learning to remove internet advertisements."
        Autonomous Agents and Multi-Agent Systems 2.3 (1999).
    """

    _filename = _get_filename("InternetAds")
    _remote_url = _get_remote_url("InternetAds")
    _length = 2_359


class DryBean(DownloadARFFGzip):
    """Dry Bean Dataset. Images of 13,611 grains of 7 different registered dry
    beans were taken with a high-resolution camera.

    * Number of instances: 13,611
    * Number of attributes: 16
    * Number of classes: 7

    **References:**

    #.  Koklu, Murat, and Ilker Ali Ozkan. "Multiclass classification of dry beans
        using computer vision and machine learning techniques." Computers and
        Electronics in Agriculture 174 (2020): 105507.
    """

    _filename = _get_filename("DryBean")
    _remote_url = _get_remote_url("DryBean")
    _length = 13_611


class Optdigits(DownloadARFFGzip):
    """Optical Recognition of Handwritten Digits (Optdigits).

    * Number of instances: 5,620
    * Number of attributes: 64 (8x8 bitmaps)
    * Number of classes: 10 (0-9)

    **References:**

    #.  Alpaydin, E., and C. Kaynak. "Cascading classifiers." Kybernetika
        34.4 (1998): 369-374.
    """

    _filename = _get_filename("Optdigits")
    _remote_url = _get_remote_url("Optdigits")
    _length = 5_620


class Frogs(DownloadARFFGzip):
    """Anuran Calls (MFCCs) Dataset. Acoustic features extracted from syllables
    of frog calls to identify the family, genus, and species.

    * Number of instances: 7,195
    * Number of attributes: 22
    * Number of classes: 4 (Families) / 10 (Species) - Typically 4 in benchmarks.

    **References:**

    #.  Colonna, J. G., et al. "Classification of frog calls using Gaussian mixture
        models and indices of bioacoustic complexity." (2015).
    """

    _filename = _get_filename("Frogs")
    _remote_url = _get_remote_url("Frogs")
    _length = 7_195


class Wine(DownloadARFFGzip):
    """Wine recognition dataset. Results of a chemical analysis of wines grown
    in the same region in Italy but derived from three different cultivars.

    * Number of instances: 178
    * Number of attributes: 13
    * Number of classes: 3

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Wine
    """

    _filename = _get_filename("Wine")
    _remote_url = _get_remote_url("Wine")
    _length = 178

class Splice(DownloadARFFGzip):
    """Primate Splice-junction Gene Sequences. The task is to recognize
    exon/intron boundaries (splice junctions) in DNA sequences.

    * Number of instances: 3,175 (full) or 1,000 (standard train subset)
    * Number of attributes: 60
    * Number of classes: 3 (reduced to 2 in many benchmarks)

    **References:**

    #.  https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)
    """

    _filename = _get_filename("Splice")
    _remote_url = _get_remote_url("Splice")
    _length = 3_190


# ============================================================
# COVID and Seagate Datasets
# ============================================================


class SeagateBinary(DownloadARFFGzip):
    """Seagate hard-disk failure detection (binary classification).

    Sensor readings from Seagate hard drives used to predict whether a drive
    will fail (1) or remain healthy (0).

    * Number of instances: 49,999
    * Number of attributes: 94
    * Number of classes: 2 (0 = healthy, 1 = failure)
    """

    _filename = _get_filename("Seagate_Binary")
    _remote_url = _get_remote_url("Seagate_Binary")
    _length = 49_999


class SeagateMulti(DownloadARFFGzip):
    """Seagate hard-disk failure mode prediction (multi-class classification).

    Sensor readings from Seagate hard drives used to predict the specific
    failure mode of the drive (11 failure categories plus healthy).

    * Number of instances: 11,800
    * Number of attributes: 94
    * Number of classes: 11 (0–10)
    """

    _filename = _get_filename("Seagate_Multi")
    _remote_url = _get_remote_url("Seagate_Multi")
    _length = 11_800


class C6StayHomeRequirements(DownloadARFFGzip):
    """COVID-19 policy dataset: C6 Stay at Home Requirements.

    Predicts the level of stay-at-home requirements imposed by a
    country/region during the COVID-19 pandemic.

    * Number of instances: 789
    * Number of attributes: 202
    * Number of classes: 3 (0 = no requirements, 1–2 = increasing restriction levels)

    **References:**

    #.  Hale, T., et al. "A global panel database of pandemic policies (Oxford
        COVID-19 Government Response Tracker)." Nature Human Behaviour, 2021.
    """

    _filename = _get_filename("C6_Stay_Home_Requirements")
    _remote_url = _get_remote_url("C6_Stay_Home_Requirements")
    _length = 789


class C7InternalMovement(DownloadARFFGzip):
    """COVID-19 policy dataset: C7 Restrictions on Internal Movement.

    Predicts the level of restrictions on internal movement imposed by a
    country/region during the COVID-19 pandemic.

    * Number of instances: 789
    * Number of attributes: 202
    * Number of classes: 3 (0 = no restrictions, 1–2 = increasing restriction levels)

    **References:**

    #.  Hale, T., et al. "A global panel database of pandemic policies (Oxford
        COVID-19 Government Response Tracker)." Nature Human Behaviour, 2021.
    """

    _filename = _get_filename("C7_Internal_Movement")
    _remote_url = _get_remote_url("C7_Internal_Movement")
    _length = 789

