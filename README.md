[travis-img]: https://img.shields.io/travis/com/makgyver/gossipy?style=for-the-badge
[travis-url]: https://app.travis-ci.com/github/makgyver/gossipy

[language]: https://img.shields.io/github/languages/top/makgyver/gossipy?style=for-the-badge
[issues]: https://img.shields.io/github/issues/makgyver/gossipy?style=for-the-badge
[license]: https://img.shields.io/badge/mit-blue?style=for-the-badge&logo=opensourceinitiative&color=8a1c06&logoColor=white
[version]: https://img.shields.io/badge/python-3.7|3.8|3.9-blue?style=for-the-badge

[![python](https://img.shields.io/badge/PYTHON-blue?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)
[![vscode](https://img.shields.io/badge/VSCODE-white?style=for-the-badge&logo=visualstudiocode&logoColor=blue)](https://code.visualstudio.com/)
[![macos](https://img.shields.io/badge/macOS-grey?style=for-the-badge&logo=apple)](https://code.visualstudio.com/)
[![open-source](https://img.shields.io/badge/open%20source-blue?style=for-the-badge&logo=github&color=123456)](https://github.com/makgyver/gossipy/)
![license]

[![Build Status][travis-img]][travis-url]
[![Coverage Status](https://img.shields.io/coveralls/makgyver/gossipy?style=for-the-badge)](https://coveralls.io/github/makgyver/gossipy?branch=main)
![version] ![issues]
[![GitHub latest commit](https://img.shields.io/github/last-commit/makgyver/gossipy?style=for-the-badge)](https://github.com/makgyver/gossipy/commit/)


# :warning: UNDER DEVELOPMENT :warning:

# gossipy 
Python module for simulating gossip learning.

## TODOs

### Features

- [ ] Logger (partially implemented)
- [ ] Multi-threading/processing
- [ ] Models cache[[Ormandi 2013]](#1) [[Giaretta 2019]](#4) (partially implemented)
- [ ] Perfect matching [[Ormandi 2013]](#1)
- [ ] More realistic online behaviour (currently it is a worst case scenario)
- [ ] Message transmission time (not sure if it can be modelled as a simple delay)
- [ ] PENS [[Onoszko 2021]](#8)
- [ ] DFL [[Liu 2022]](#10)
- [ ] Segmented GL [[Hu 2019]](#5)
- [ ] CMFL [[Che 2021]](#9)

### Extras

- [ ] Documentation
- [ ] Add 'Weights and Biases' support

## References
<a id="1">[Ormandi 2013]</a>
Ormándi, Róbert, István Hegedüs, and Márk Jelasity. 'Gossip Learning with Linear Models on Fully Distributed Data'. Concurrency and Computation: Practice and Experience 25, no. 4 (February 2013): 556–571. https://doi.org/10.1002/cpe.2858.

<a id="2">[Berta 2014]</a>
Arpad Berta, Istvan Hegedus, and Robert Ormandi. 'Lightning Fast Asynchronous Distributed K-Means Clustering', 22th European Symposium on Artificial Neural Networks, (ESANN) 2014, Bruges, Belgium, April 23-25, 2014.

<a id="3">[Danner 2018]</a>
G. Danner and M. Jelasity, 'Token Account Algorithms: The Best of the Proactive and Reactive Worlds'. In 2018 IEEE 38th International Conference on Distributed Computing Systems (ICDCS), 2018, pp. 885-895. https://doi.org/10.1109/ICDCS.2018.00090.

<a id="4">[Giaretta 2019]</a>
Giaretta, Lodovico, and Sarunas Girdzijauskas. 'Gossip Learning: Off the Beaten Path'. In 2019 IEEE International Conference on Big Data (Big Data), 1117–1124. Los Angeles, CA, USA: IEEE, 2019. https://doi.org/10.1109/BigData47090.2019.9006216.

<a id="5">[Hu 2019]</a> Chenghao Hu, Jingyan Jiang and Zhi Wang. 'Decentralized Federated Learning: A Segmented Gossip Approach'. https://arxiv.org/pdf/1908.07782.pdf

<a id="6">[Hegedus 2020]</a>
Hegedűs, István, Gábor Danner, Peggy Cellier and Márk Jelasity. 'Decentralized Recommendation Based on Matrix Factorization: A Comparison of Gossip and Federated Learning'. In 2020 Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2020, pp. 317-332. https://doi.org/10.1007/978-3-030-43823-4_27.

<a id="7">[Hegedus 2021]</a>
Hegedűs, István, Gábor Danner, and Márk Jelasity. 'Decentralized Learning Works: An Empirical Comparison of Gossip Learning and Federated Learning'. Journal of Parallel and Distributed Computing 148 (February 2021): 109–124. https://doi.org/10.1016/j.jpdc.2020.10.006.

<a id="8">[Onoszko 2021]</a>
Noa Onoszko, Gustav Karlsson Olof Mogren, and Edvin Listo Zec. 'Decentralized federated learning of deep neural networks on non-iid data'. International Workshop on Federated Learning for User Privacy and Data Confidentiality in Conjunction with ICML 2021 (FL-ICML'21). https://fl-icml.github.io/2021/papers/FL-ICML21_paper_3.pdf

<a id="9">[Che 2021]</a>
Chunjiang Che, Xiaoli Li, Chuan Chen, Xiaoyu He, and Zibin Zheng. 'A Decentralized Federated Learning Framework via Committee Mechanism with Convergence Guarantee'. https://arxiv.org/pdf/2108.00365.pdf

<a id="10">[Liu 2022]</a>
Wei Liu, Li Chen and Wenyi Zhang. 'Decentralized Federated Learning: Balancing Communication and Computing Costs'. https://arxiv.org/pdf/2107.12048.pdf
