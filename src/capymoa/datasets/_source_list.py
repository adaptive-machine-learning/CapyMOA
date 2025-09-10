"""This is a module responsible for storing the URLs of the datasets."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _Source:
    arff: str
    """The URL of the ARFF file. Can be optionally compressed."""
    csv: Optional[str]
    """The URL of the CSV file. Can be optionally compressed."""


# DROPBOX: When downloading from Dropbox, the URL parameter `dl=1` must be set
# to force the download.
SOURCE_LIST: Dict[str, _Source] = {
    "Sensor": _Source(
        "https://www.dropbox.com/scl/fi/tyvtg4g9nop80gc2zuhog/sensor.arff.gz?rlkey=dfocd19pu67lvp4zj2tcaajoq&st=ajjxsxg3&dl=1",
        "https://www.dropbox.com/scl/fi/aloisxhd7p50f669b9rxe/sensor.csv.gz?rlkey=ndiscm2qv4cvgnf5r0ly8ikjg&st=mp2gok4c&dl=1",
    ),
    "Hyper100k": _Source(
        "https://www.dropbox.com/scl/fi/9mmfc0kj8hvl60at1xi4b/Hyper100k.arff.gz?rlkey=gw1yfdg5wfyj5d2dswdvy0ksr&st=w8cyz8dd&dl=1",
        "https://www.dropbox.com/scl/fi/il9cbzqqtinx9ygft7zg8/Hyper100k.csv.gz?rlkey=w34bl341pu6e7wzugelqpxil7&st=nng0hnj5&dl=1",
    ),
    "CovtFD": _Source(
        "https://www.dropbox.com/scl/fi/syuuteps87v84p9imwc2u/covtFD.arff.gz?rlkey=ojznbqhbrcnser4z70tn8vc4a&st=i4o0hlsg&dl=1",
        "https://www.dropbox.com/scl/fi/tmhbnmhoyn86xj4oiurff/covtFD.csv.gz?rlkey=0e97cm7erkus7g1nej41785ek&st=8xfl6gkm&dl=1",
    ),
    "Covtype": _Source(
        "https://www.dropbox.com/scl/fi/kwjvr5kn0l0u5l4gd5788/covtype.arff.gz?rlkey=6vlomqdoi3oud26o1ngyjoibr&st=5jvy1ctv&dl=1",
        None,
    ),
    "RBFm_100k": _Source(
        "https://www.dropbox.com/scl/fi/qwes0rxf4dg3c6vu2867b/RBFm_100k.arff.gz?rlkey=hfkz3k1lir85bag1ha9az667e&st=8kybyd98&dl=1",
        "https://www.dropbox.com/scl/fi/zezfr5xs6de8vgwh1ilxm/RBFm_100k.csv.gz?rlkey=i7skqltc67iffig27foy7w74b&st=hhshaxu1&dl=1",
    ),
    "RTG_2abrupt": _Source(
        "https://www.dropbox.com/scl/fi/4tx564vnqqm88hieu0lpv/RTG_2abrupt.arff.gz?rlkey=7ap9l7721g5lrgppobi3if5n9&st=4225qqnt&dl=1",
        "https://www.dropbox.com/scl/fi/3qd7wjaakxbi5209i369d/RTG_2abrupt.csv.gz?rlkey=5s2sisrsj8txmu9sh2rvkqoon&st=1u4qr4xe&dl=1",
    ),
    "ElectricityTiny": _Source(
        "https://www.dropbox.com/scl/fi/kjce4npqbqztryp2fcsa7/electricity_tiny.arff.gz?rlkey=ktcad35l6sdaeit2v4rlpw1hc&st=u7mdhpvl&dl=1",
        "https://www.dropbox.com/scl/fi/y56mumnq4y4wcvkug54gd/electricity_tiny.csv.gz?rlkey=lahzupoh3mq8jnxvwgnha4roj&st=40cj79qp&dl=1",
    ),
    "Electricity": _Source(
        "https://www.dropbox.com/scl/fi/p7btran9xdas20neva917/electricity.arff.gz?rlkey=xt8hc6542b8wr01eitcw7mwpz&st=paqojdax&dl=1",
        "https://www.dropbox.com/scl/fi/n6vswhxz8i8f5jd8nc7bu/electricity.csv.gz?rlkey=oczanfxru37xfk0xn3qf78emh&st=h7vjpwr8&dl=1",
    ),
    "CovtypeTiny": _Source(
        "https://www.dropbox.com/scl/fi/ganjs6pfdzwhqtw1ijq8j/covtype_n1000.arff.gz?rlkey=acezd1qj967a8fcinsnlwixyk&st=7ms3jt9o&dl=1",
        None,
    ),
    "CovtypeNorm": _Source(
        "https://www.dropbox.com/scl/fi/g63sw31cqykylv2e07cnd/covtypeNorm.arff.gz?rlkey=gdhb3zhtxd8c4djew8vf14saa&st=aoii1nk7&dl=1",
        "https://www.dropbox.com/scl/fi/mnr5w39jh5v7urpznrcvu/covtypeNorm.csv.gz?rlkey=4pz0cd3dx0ab0t07rdfltrmpq&st=kbomo6q2&dl=1",
    ),
    "Fried": _Source(
        "https://www.dropbox.com/scl/fi/rvlr6lo4k1ryeelnrl61t/fried.arff.gz?rlkey=5pnnp6ixdp3lz4k88r2mvmxu1&st=wneq1abb&dl=1",
        "https://www.dropbox.com/scl/fi/zo9zdar7zbj5b582nirni/fried.csv.gz?rlkey=f44975aigjminfbuwp1cfe5xu&st=vvudbcwx&dl=1",
    ),
    "FriedTiny": _Source(
        "https://www.dropbox.com/scl/fi/ck7mhvljj6oxb7e0bn26k/fried_tiny.arff.gz?rlkey=jmw5f613j52wqurih18v5rb10&st=nre84p53&dl=1",
        "https://www.dropbox.com/scl/fi/ejmu51dwu5u0d3z94f2vq/fried_tiny.csv.gz?rlkey=31miahxhab5g0jstpe9hcbe40&st=5y6lrdxa&dl=1",
    ),
    "Bike": _Source(
        "https://www.dropbox.com/scl/fi/r3lmdpzhyzv2h0mdh4802/bike.arff.gz?rlkey=9eq3x5onmgtpmhi1kpjs25u4c&st=ikx0urzf&dl=1",
        "https://www.dropbox.com/scl/fi/srwuiua429eqf4um750z0/bike.csv.gz?rlkey=6e537xxjvs2hcaev73fkpc4yy&st=2gebrfhf&dl=1",
    ),
}
