# CHANGELOG



## v0.10.0 (2025-06-09)

### Ci

* ci: skip slda tests on ci for time being ([`9cb4a92`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9cb4a92095c72ad670622b9426454fab458ef029))

* ci(release): update test steps ([`39b25d7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/39b25d789f0f5c21b476af1ea1eed7aa72a6b819))

* ci(tasks.py): propagate errors for CI/CD pipeline (#264) ([`6ed24d7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6ed24d79ce42a53fd978c2855c34796410ec998c))

* ci: update pr workflow (#257) ([`9647866`](https://github.com/adaptive-machine-learning/CapyMOA/commit/964786607cc3dec3cca7c9a680fbe44fc47de0ac))

### Feature

* feat(anomaly): streaming isolation forest ([`de49daa`](https://github.com/adaptive-machine-learning/CapyMOA/commit/de49daaa9049c4c264e5929715068f8625a10ee0))

* feat(ocl): improve batch support and add built-in ANN (#267) ([`4427e9f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4427e9f9053e540036e2863a55a71fcc24f8a60e))

* feat(ocl): add NCM and SLDA ([`05ed2e8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/05ed2e83ce48300c951390494a77e6ca7572e8bc))

* feat(ocl): add SplitCIFAR100ViT and SplitCIFAR10ViT ([`e337bde`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e337bde324c8a0abfb0566fc71ecb1d30c1ff937))

* feat(BatchClassifier): improve mini-batch processing performance ([`0a8cea5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0a8cea5a45d2f93b7c4add371d714e2dfea8fbb1))

* feat(classifier): add PyTorch batch finetune learner (#259) ([`4b40350`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4b40350272dd2f48ba70c026e8e7971a9a25ed29))

* feat: streaming random histogram forest for anomaly detection ([`7ada724`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7ada72454ea314ddff042cfe364adf0c4b3a7113))

* feat(ocl): add prequential evaluation to ocl (#250) ([`e1669fe`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e1669fe1ce910d35584635953fcf35fd5fcf9869))

### Fix

* fix: remove debug print statement ([`52d0a6a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/52d0a6ad2fa03c70110ba2555b46eb1ac0d094ce))

* fix: remove debug print statement ([`ebb17d2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ebb17d232ddc1f6be3631615babee3724f7ddb42))

* fix: update example ([`ab3b2f8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ab3b2f89759610c2663f10f297b942c01d25e751))

* fix: correct import path ([`1c7fa0f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1c7fa0f50051e5ea2e70cf70dfc47ff46c903d89))

* fix: remove unused imports ([`6129d16`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6129d1644fe675835d9b708d16d24ddb54c5f7d9))

* fix: improve code formatting with ruff ([`7c41b9f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7c41b9f71a4fe3d47f88b6894f1987821dcb3cb7))

* fix: correct anomaly score interpretation and type hint in StreamRHF ([`f792266`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f7922666186f164d92a79a9d9cf655704c884296))

### Performance

* perf(instance): speed up copy to java instance ([`02dccca`](https://github.com/adaptive-machine-learning/CapyMOA/commit/02dcccaec2552c22cc0716aa59090d36af888559))

* perf(ocl): add PyTorch dataset preloading ([`d91c36b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d91c36ba589e8059b6a1b594ce0022673b8071eb))

### Refactor

* refactor(ocl): change project structure ([`abb6458`](https://github.com/adaptive-machine-learning/CapyMOA/commit/abb645873fcea0b09579dcc75ef1f7863751f178))

### Unknown

* Revert &#34;Drift detector evaluation v1 + STUDD + fix HDDM&#34; (#258) ([`7888975`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7888975d688fe4fb61d3369495720001fb50228f))

* fix eval detector ([`559fb54`](https://github.com/adaptive-machine-learning/CapyMOA/commit/559fb54ce45b63ec75b4549028e3e82544fda81d))

* reset detector fix ([`bcc9c56`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bcc9c56918b04a5758fcd151917f0924a28ee12c))

* fix hddm w ([`93b7796`](https://github.com/adaptive-machine-learning/CapyMOA/commit/93b7796b85c370d73946d629530aca5d06704f26))

* updated drift evaluator + studd ([`0f3e2ef`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0f3e2ef83d34f14f325bd5d5d7940c6a00affb48))


## v0.9.1 (2025-05-08)

### Chore

* chore(version): increment version to 0.9.1 ([`c2db396`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c2db39611bdb383ed6e4d02790e7b85b198a9852))

### Ci

* ci: fix temporary directories for windows (#255) ([`f030afb`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f030afb1059a864a0d04a86220f8e3bffdabbef9))

* ci: use ``python -m`` to fix windows release (#254) ([`85fe5f5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/85fe5f560cb7f1aeec62ad756b57700852a49c9a))

### Documentation

* docs: add license to pypi (#253) ([`1c9bc4b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1c9bc4bf714e2a9eadd440967123feca49bb856f))

### Fix

* fix(stream): stop ``CSVStream`` rounding ``y_value`` erroneously (#252) ([`82d66ec`](https://github.com/adaptive-machine-learning/CapyMOA/commit/82d66ec7763af45435eb90529c6378cfce7ec7dd))

* fix(stream): fix ``get_moa_stream`` ``ValueError`` (#251) ([`effd818`](https://github.com/adaptive-machine-learning/CapyMOA/commit/effd8188321fdd99c3cd3f8ce8c3a84393f54aa9))


## v0.9.0 (2025-03-28)

### Chore

* chore(version): increment version to 0.9.0 ([`77a333f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/77a333f80f602e3490cd682c7af7977df648ed0d))

### Ci

* ci: add ``pr.md`` explaining CI ([`8561add`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8561add09a14d9518ef049cbe6538de8258330d4))

* ci: set sphinx 8.1.3 and add ruff target version ([`0895e32`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0895e32b2f24ead141b79a20075c72161b517d5d))

* ci: add ruff formatter and linter (#216) ([`f4775c1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f4775c1bca7e520932f1eb2d518eff8df5da7b60))

### Documentation

* docs(api): add missing packages to api docs (#239) ([`da059a2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/da059a24cf252bb03510d3cce22bacc7c1b24a9c))

* docs: adding bibtex to readme, website, and cff file ([`34ab916`](https://github.com/adaptive-machine-learning/CapyMOA/commit/34ab9162e8b3c176a1f7f998b82c3c656f7bd07a))

### Feature

* feat: update ``moa.jar`` (#244) ([`8f044c0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8f044c01db3c5a974d480d06f44afb8b4ae79c9a))

* feat(ocl): add optional intra-task evaluation (#242) ([`42ab860`](https://github.com/adaptive-machine-learning/CapyMOA/commit/42ab860a60a6ef0b449c21502392ff9296729c21))

* feat: add ocl tutorial and batch base classes ([`0eec6c6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0eec6c6462d4e90f602cb46fe2860aae542e842f))

* feat: add ocl eval loop (#236) ([`e7544a0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e7544a06794898e01bae388e2c2a923473981b80))

* feat: update srp with minibatch function (#227)

Updated the SRP wrapper with mini-batch function and associated moa.jar ([`e4b6ab9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e4b6ab99388d2ca317b7abf57131667f2ee16510))

* feat: add ocl streams ([`7fe5d33`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7fe5d33ebfb5fba927b8526d9560b6ce1cb93e21))

* feat: update anomaly score to increase with abnormality (#220)

Co-authored-by: Justin Liu &lt;justinuliu@gmail.com&gt; ([`122a7e3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/122a7e380d37eeb90c5a033812c5fb5904a292a5))

### Fix

* fix: remove broken typecheck in ``save_model`` for windows (#246) ([`ea6f24e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ea6f24e75422b59ee41d82cdc7235f1b5ef77185))

* fix: add getter functions to Schema (#241) ([`7a5aa36`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7a5aa366cb2a3e950eb284bc9e92592e9eb77ad9))

* fix(OnlineIsolationForest): add random generator to assign unique seeds to OnlineIsolationTrees ([`dbb95f5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dbb95f57f2ec0d4e6fbeea6cb13b04121cfb40ad))

* fix: formatting ([`40d75e9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/40d75e920af2501132695380610f1d599408a83b))

* fix: fix logic for multi-threads ([`75237b9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/75237b9d59722476f74f3ee73ab88a6fc542c463))

* fix: updating .cff file for correct citation information (#232) ([`1987a19`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1987a192f922f8d02c3141881e66e306e7ac43a0))

* fix: leveragingbagging parameter configuration  (#203) ([`5c67fc9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5c67fc994eea9723059f5f49b4ead48a61832da1))

* fix: increasing title underline length ([`7b324e2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7b324e2a5a9688a3a448b17cb2c7cf14e7ad0a67))

* fix: update jpype to v1.5.1 ([`86d4d9d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/86d4d9d8f33a0ee94208156c67c23b1c665db5af))

### Refactor

* refactor(test_anomaly_detectors): update accuracy output value of OnlineIsolationForest ([`9d85d21`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9d85d21e53d934a4333d57fa134d4d9f0efe0cd5))

* refactor: use python iterator instead of `next_instance` ([`b16268c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b16268c064ea9c348800263df844e60afc7208c5))

* refactor(stream): use pythonic iterators for streams ([`4ba4dd9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4ba4dd9c31796fbeb3d2b339f1bdf852e0e55521))

### Style

* style: format to fix quality checks (#228) ([`18321ff`](https://github.com/adaptive-machine-learning/CapyMOA/commit/18321ff9a7c547caa7d47f6e794c8632d1aeb046))

### Test

* test: skip non-ASCII moajar test on Windows (#245) ([`c9c49db`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c9c49dbc13d0a99090577a28e1b5c8ebc6bdfd8e))

* test: fix test workdir and windows test import (#243) ([`fe40df1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/fe40df178d042f57af45fd2f23f4dac10411d963))

* test: update progrss bar test (#234) ([`a931af6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a931af6e023966cf8c6e059d6fc7f767f1735d0d))

### Unknown

* doc: updating collaborator name

Fixing Justin&#39;s Name in the Bibtex
Authored by Yibin Sun ([`befae5b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/befae5bc85367ef57f266c553848a0105d11ddef))


## v0.8.2 (2024-12-04)

### Chore

* chore(version): increment version to 0.8.2 ([`6253bff`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6253bffda2e24c6e6ca2f4ee9c79a64f8f2fcae2))

### Fix

* fix: fix release.yml ([`6be2832`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6be2832f586bfd6cfab3537235ab65bb4cecb109))


## v0.8.1 (2024-12-04)

### Chore

* chore(version): increment version to 0.8.1 ([`49ae3ec`](https://github.com/adaptive-machine-learning/CapyMOA/commit/49ae3ecfe09e06104dd167e266b0e85f1fa0b38f))

### Fix

* fix: update docker release ([`af93fca`](https://github.com/adaptive-machine-learning/CapyMOA/commit/af93fcafe907a367f9f29079722d17bd50b99c01))


## v0.8.0 (2024-12-01)

### Build

* build: use python 3.11, 3.12, and 3.13 ([`993b12a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/993b12a588fbff88927f783dd6f9ed83c42a87c5))

### Chore

* chore(version): increment version to 0.8.0 ([`6656242`](https://github.com/adaptive-machine-learning/CapyMOA/commit/66562427eb3131a2293e187b11a6f0c09cc65b5b))

* chore: updating moa jar 24-09-08 ([`b7e958c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b7e958c9acd0a25066fa61761300c31c5f793290))

### Ci

* ci: fix docker triggers ([`2759800`](https://github.com/adaptive-machine-learning/CapyMOA/commit/275980060a9713a32529441e32a3b2cae36a07a4))

* ci: fix automated docker versioning ([`0cc493e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0cc493e618b13365be0699753ff760801a67fe6e))

### Documentation

* docs: add autoML notebook with examples (#198) ([`ac7e578`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ac7e578fc1be2186d8485c40bfdd7bae1b574102))

* docs: update csmote init doc ([`0963190`](https://github.com/adaptive-machine-learning/CapyMOA/commit/09631906139b9aa19a67644693f2fa287d294d7a))

* docs: fix broken link in README ([`40e98d0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/40e98d0cf782a168a479ceaef950152fbb12becf))

### Feature

* feat: add optional progress bars ([`0610c4b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0610c4bd205fb0e1bb988886e3edee65143bd420))

* feat: abcd, a drift detector for univariate and multivariate data ([`5c75c24`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5c75c2445e55e6f85ef9380fb7cc1e52436d3128))

* feat: add shrub ensembles for online classification

Buschjäger, S., Hess, S., &amp; Morik, K. J. (2022, June). Shrub ensembles for
online classification. In Proceedings of the AAAI Conference on Artificial
Intelligence (Vol. 36, No. 6, pp. 6123-6131). ([`f966e32`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f966e32afa080dffca364978cfcb0c76d8ceb9a7))

* feat: added wrapper code for 6 stream generators ([`ee2cd30`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ee2cd305d49f801281b5913fc649b5fea0202372))

* feat: add autoclass for automl ([`b5adc95`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b5adc950436f37b8fc0f966d305791d85ea8261c))

* feat: added code for led and agrawal stream generators ([`3356f00`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3356f0070c66ca08eaabdab8b557a65ea7b2254e))

* feat: added drift object capabilities to arf ([`f792667`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f792667754c68b26a95a6f08480ac6dbc5b9e3cf))

* feat: added weightedknn ([`e9c5757`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e9c57572808bb465ebc3b2ea7ae553c7c65ea29c))

* feat: create synthetic data stream for anomaly detection

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`d1dbd78`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d1dbd78856fcd8db1e06039adda604fc7854bb69))

* feat: clustering, 4 wrappers, notebook and documentation ([`b5c8f28`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b5c8f28501d67956a326c328bf584951e2a4a9ed))

* feat: clustering methods, documentation and notebook ([`dc68be6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dc68be63907cc74a88c84be679bc51d0d232d15a))

### Fix

* fix: fix numpy 2.0.0 compatibility (#199) ([`7bbaa96`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7bbaa963950cdb19ad640ccb8d26fcd68bf50b24))

* fix: ensuring deterministic behavior for driftstream ([`33ba7ba`](https://github.com/adaptive-machine-learning/CapyMOA/commit/33ba7ba81a3af291dd687640f8c4dc16dabd187b))

* fix: fixing documentation for notebooks toctree ([`21fb4aa`](https://github.com/adaptive-machine-learning/CapyMOA/commit/21fb4aa2b7abaac0a9c0f50f4d7ec3ed5c248762))

* fix: adding gifs shown in notebook ([`6c08d39`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6c08d3963b33baaf405fd75305f87c21e3b5d7b0))

### Test

* test: stop skipping notebooks in CI/CD pipeline ([`dfcd0c6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dfcd0c614bc1e4273ad0a8355368d18640ebbcfa))

### Unknown

* Pipelines V2 (#200)

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`09a80eb`](https://github.com/adaptive-machine-learning/CapyMOA/commit/09a80ebc5d8681a42fca9d1d97365d68f97ab554))


## v0.7.0 (2024-08-03)

### Chore

* chore(version): increment version to 0.7.0 ([`d145644`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d1456442c5c7d0be4b95029c0913aa0c8b8b8673))

### Ci

* ci: fix docker versioning ([`bbfc8b5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bbfc8b5f693e920c72e67e75d69133a28eb074b7))

### Documentation

* docs: update plotting to prediction interval notebook ([`f33c3d1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f33c3d13ea8b3f7c4c01df4a1692432614e8094d))

### Feature

* feat: clustering base classes with simple visualization

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`09b3a60`](https://github.com/adaptive-machine-learning/CapyMOA/commit/09b3a60f5a8c1146efd69e654a2211b5928c30ed))

### Fix

* fix: update visualization for compatibility ([`e25f6b6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e25f6b6eddee42666c54d4f39fff823d085b6409))


## v0.6.0 (2024-07-31)

### Chore

* chore(version): increment version to 0.6.0 ([`bed6fe0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bed6fe066f8a80a732a745b9393b8efd91f2ed1c))

### Ci

* ci: fix workflow trigger of docker build ([`d8fdd55`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d8fdd557e2c2a39c2bfefdf6be03c2e77d63e209))

### Feature

* feat: add ``restart_stream=False`` as an option in evaluators ([`a965dc4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a965dc45c4d2edf796c76064ba2c880824c477d1))

* feat: add clustering and ssl ([`ae8b592`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ae8b592ae658a7dcbf6c314e0764bb51e36afc5d))

* feat: add IJCAI 2024 tutorial ([`cac7900`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cac790098f6eb62dd83e787b788a105afef1b094))

### Fix

* fix: remove tutorial notebook ([`d82e05d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d82e05d2c9a5cf2b5e757636b3642cf3bbf10c00))

* fix: add overwritten changes in commit cc13390 for recurrent concept drifts ([`472a597`](https://github.com/adaptive-machine-learning/CapyMOA/commit/472a597cd4c18b7cb84884bc1acd9821a3ae24ca))


## v0.5.0 (2024-07-30)

### Chore

* chore(version): increment version to 0.5.0 ([`dc69780`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dc69780b189ed468adefde9924f3d12745220bd0))

### Ci

* ci: fix bad condition ([`b096339`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b0963396e749d714e57dd0aa56c57b6807f167f2))

* ci: trigger docker builds on release ([`1ef2ad6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1ef2ad6ba1c9efd83c84882f8608040fae333330))

### Documentation

* docs: recommend https git clone ([`5c9715a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5c9715ad6ffb33a1d85cb80e41220166f306bdd7))

* docs: fix ``__init__`` documentation ([`29c516a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/29c516a5ae1d48b37ac506b006ff3a78e9900179))

* docs(about.rst): add Justin Liu ([`4d97ead`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4d97ead6012c659d3c8eed1f6c9af78b2e413019))

### Feature

* feat: add reference to Autoencoder anomaly detector ([`d8787e8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d8787e80d4d1dc5527ca5c47c88743751760d6f9))

* feat(Anomaly-Detection): add Autoencoder anomaly detector ([`e27cbe9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e27cbe98c8736d161304e6bb77e8c6918b3bbb91))

### Fix

* fix: updating AD docs and notebook ([`47ee056`](https://github.com/adaptive-machine-learning/CapyMOA/commit/47ee056ae7d2c8f18c2be30f7942b2bec63936d8))

* fix: corrections to anomaly evaluation ([`57b93c3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/57b93c3d599a0c6a16a0f822b1060b63456648fd))


## v0.4.0 (2024-07-25)

### Build

* build: ignore the stub build issue in editable installs ([`347ed74`](https://github.com/adaptive-machine-learning/CapyMOA/commit/347ed74dbe4ef2c523a5ed784273b2d5831adc27))

### Chore

* chore(version): increment version to 0.4.0 ([`6e423cc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6e423cccf7bf30c295fa18608e7d2540bb88648b))

### Documentation

* docs(tutorial.rst): add talks section ([`c5a339b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c5a339b9b8188bceea46bb5393a337e55d03539a))

* docs(drift): update drift detector docs ([`4c69478`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4c6947875c31979fe7f04e592ebeb31e6951d63d))

* docs: rebase ([`748cc94`](https://github.com/adaptive-machine-learning/CapyMOA/commit/748cc9433dc48dd2d99fdab834cac1632594ee45))

* docs: use ``sphinx.ext.autosummary`` to generate more complete api docs ([`01d270f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/01d270f25c69a37917b8a184758d60e338754b08))

* docs: add about us page ([`f3af4d5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f3af4d5ce08caa2c04a05901ebc4f644334ea16b))

### Feature

* feat(Anomaly-Detection): add cumulative and prequential evaluation for anomaly ([`a8124bc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a8124bc5d2ce4abe0191a714ab5f5199d04f8a55))

* feat: updated evaluation

Co-authored-by: Spencer Sun &lt;spencer@Spencers-MacBook-Pro.local&gt;
Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`cc13390`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cc133900b2553fd3bdc1afac3a3961ddd5594259))

* feat: add recurrent concept drift API ([`24327fc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/24327fcb0f9df15fa9b90f1a8234561a57d6d3a6))

* feat: drift detection API

Co-authored-by: Vitor Cerqueira &lt;cerqueira.vitormanuel@gmail.com&gt; ([`50d76ad`](https://github.com/adaptive-machine-learning/CapyMOA/commit/50d76ad9b9ae8dd6b4725b8cbc28f5367d498a8b))

* feat(anomaly): add ``OnlineIsolationForest`` anomaly detector ([`5a57bb0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5a57bb024e5213bcbb5ca25f73b681c84d856d0f))

* feat: add anomaly_threshold and size_limit parameters to HalfSpaceTree (#135)

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`21804d0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/21804d093661f5d628d5869716d8d9a15f72d20d))

* feat(Misc): add unit tests for model save and load functionality ([`739f8af`](https://github.com/adaptive-machine-learning/CapyMOA/commit/739f8af3ac9044896b3f6b7307bd120004c59ae8))

### Fix

* fix: updating evaluation to use prequential results ([`2b6abab`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2b6abab0278215700ae6f7fdcc2f850b39cb2a19))

* fix: fix issue saving and loading models ([`04d9207`](https://github.com/adaptive-machine-learning/CapyMOA/commit/04d9207b6cdda17a190117057335a662a189ec93))

### Test

* test: speed up ``parallel_ensembles.ipynb`` ([`e9a6bb0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e9a6bb0ad634537fa96a21fa5f681292db39080e))

* test: rename name of unit test ([`4dbecb3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4dbecb3ed47573b34ca2411b3ea74ca427f5df68))


## v0.3.1 (2024-06-10)

### Chore

* chore(version): increment version to 0.3.1 ([`c065d18`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c065d183695944793243475bb88a6ed8caf8b43a))

### Fix

* fix: updating tutorials formatting ([`36a0a29`](https://github.com/adaptive-machine-learning/CapyMOA/commit/36a0a29d006036bacbacb1215ec0665b86b9c499))


## v0.3.0 (2024-05-30)

### Build

* build: add gcc and g++ to docker for arm support ([`3ede9cd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3ede9cd07c1c45f5cace514bec16b9c41e5c7182))

* build: `build.clean-moa` now includes `build.clean-stubs` ([`afcc080`](https://github.com/adaptive-machine-learning/CapyMOA/commit/afcc0808266cf2671c2dafc61f208936bf961525))

* build: fix dockerfile base ([`88b7b19`](https://github.com/adaptive-machine-learning/CapyMOA/commit/88b7b196f3b5392192ea4f696e656b8e292d1ddf))

* build: add docker arm64 ([`7f1d25f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7f1d25fe5ec24d7da7a93678bb31a86a5d18f177))

* build: add docker ([`e07288c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e07288c177043a9c6a4ccc0b15cf00592e02933c))

### Chore

* chore(version): increment version to 0.3.0 ([`5029484`](https://github.com/adaptive-machine-learning/CapyMOA/commit/50294840a608f5d065bcadf62899055fc9e13aa2))

### Documentation

* docs: adding documentation for regressors ([`64c7105`](https://github.com/adaptive-machine-learning/CapyMOA/commit/64c71059fecf5a062a5b771e3c220c3f77686fc8))

* docs(Anomaly-Detection): add anomaly detection in the extra tutorials section ([`3bc6f94`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3bc6f94bff1638be08da8de78ce660c1a04ece1c))

* docs: fix benchmarking link ([`06202f8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/06202f8b6067afbe42bc103ad7524f28d078b480))

* docs: update installation.md ([`9324fd4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9324fd4c7a8c0ef742498bd07894a731153c909e))

### Feature

* feat: parallel ensembles (#132)

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`1a3e6a3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1a3e6a3fa182327318f21570536151ce0a9bd7f4))

* feat(Misc): export and import models ([`a1af686`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a1af6864686a2cbb3a883e2c48f2385c2b3be74a))

* feat: adding LeveragingBagging and OnlineAdwinBagging classes ([`25f0f75`](https://github.com/adaptive-machine-learning/CapyMOA/commit/25f0f757773acb7dd892bb3628bbbc7897cfbc80))

* feat(pipelines): updated moa jar and minor changes ([`e71b065`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e71b065fb45409654e6fc26b3b465d7a37efb7f0))

* feat(pipelines): updated notebook introducing the concepts ([`4c787d6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4c787d6991e099a7b42c5d1772df8b2adc226e4e))

* feat(Anomaly-Detection): add abstract class for anomaly detector algorithms ([`df74b25`](https://github.com/adaptive-machine-learning/CapyMOA/commit/df74b25c415279dba39c42d760422b97c4512460))

* feat(STREAM): change the logic when dealing with csv input ([`2c2a89d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2c2a89dd2eb00bee4063ab58ca634edd6d0d3319))

* feat(CSMOTE): adding CSMOTE for imbalanced learning ([`e39d006`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e39d00635b4136c0ed57da56ec086f8a23843cdb))

* feat(Anomaly-Detection): add anomaly detection notebook ([`f0b6d4f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f0b6d4f6e40551f1bca2b75f683145b7818cff51))

* feat(Anomaly-Detection): add Half-Space Trees

Add wrapper for Half-Space Trees for MOA. ([`aa85c0f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/aa85c0f6f1719baa1ea9968217281d379e4b5352))

* feat: evaluation tests and fixes ([`147612b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/147612bd3109d856bdf7c9b12621e783b777bb0f))

* feat: update plotting regression/prediction interval ([`bf3f7ce`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bf3f7cef441f54d1c4d3e72e9c015989cf8c969a))

* feat: add HAT, DWM, and SAMkNN wrappers ([`314db2e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/314db2e79ebba487e677a46fabcf040dbfb58ad2))

* feat: add plot_regression/prediction_interval_results functions ([`fe44e0f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/fe44e0fbbf963cd07d1a0d87c93b929b3e8c903f))

* feat: add wrapper for hyperplane (c and r) generators ([`6cd23af`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6cd23af8251494c675e1ea824d2699e0aaebc5db))

* feat: add prediction interval notebook ([`f64a2e1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f64a2e19a60d81c0c0413e854d92e78da750c3e1))

### Fix

* fix: updating save_load notebook ([`c4e11b0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c4e11b00f4d15eede58688ef5b73057183caffe7))

* fix(Misc): add a title for tutorial of save and load model ([`eaad513`](https://github.com/adaptive-machine-learning/CapyMOA/commit/eaad513bdf949b95b8a762050617698277bac82a))

* fix: adding tests for new ensembles ([`7ecd80d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7ecd80d7e2dbbf1980e007f0502b17cf6a0a59ee))

* fix: expanding example notebook ([`105bf4c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/105bf4cdf5a52d5733a47bb74bad61b88eb4c85a))

* fix: add notebook 07 to toctree ([`d527e29`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d527e291177e734322f7d2e578e70d3a5f003a8b))

* fix: moajar hash ([`e02d8ac`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e02d8ac7ce8d2934d206a0c28fd5d453dbcef5f5))

* fix: add pipelines notebook to invoke.yml ([`800ed73`](https://github.com/adaptive-machine-learning/CapyMOA/commit/800ed73b87173d6877250045cc071a0c9b3bdf26))

* fix: call prepareForUse() in initializer of stream when schema is provided ([`9f29e61`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9f29e61847481c4da36cca60d3fb7054b6479aca))

* fix: corrected docstring for stream_from_file ([`e627f2d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e627f2d3b4c498841c3891d0c43b24e542491220))

* fix: updated doc and ARFF exception handling ([`187c429`](https://github.com/adaptive-machine-learning/CapyMOA/commit/187c429a8d07c12ffa6d890674b1eb897abe5327))

* fix: fix stream consistency ([`f363788`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f3637882f9ed5d29037179de47d72f9e42a17da7))

* fix: force target as int when categorical ([`bc61941`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bc6194197accb6ba48a6517a9e251eca56878936))

* fix: update from_custom function in _stream.py ([`1bbba9b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1bbba9b4b812d04437032795500d78779770cb92))

* fix: update PytorchStream.py with target_type ([`d8e9446`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d8e94460a671f4298e10b7fb44165a64119f2a2e))

* fix: update instance.py for target_type ([`e602dc3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e602dc3e48290b2b6da3f0ed986aad3eb7cc9f08))

* fix: update test_batch to enforce classification in basic test ([`bda30f5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bda30f530931ab5217b182a17ff3d656752cd480))

* fix: small alteration ([`d8c44d8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d8c44d8c9045751dd9bb99e2fa745673fb290558))

* fix: remove None| and update test_regressor ([`2cc3ba5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2cc3ba51030d1cf9676f3a9a9dfd7fed8e046c6d))

* fix: refine documentation for SOKNl and SOKNLBT ([`1c013f0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1c013f0db2b1054eb33c191333769a1f4c4d5aec))

* fix: fix the end-strings ([`bf114cf`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bf114cf0559371d023c03f8bcaf7d5518bfa71ed))

* fix: update evaluation.py ([`bf62b0e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bf62b0ed2d9cd6dda03afb683ecfa284cc43df97))

* fix: update evalutation.py ([`6e1e90f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6e1e90f73d65c33a62c71a3638b203445301b7a1))

* fix: update moa-url and sha256 value ([`491f617`](https://github.com/adaptive-machine-learning/CapyMOA/commit/491f617275446467ab613a0c0eeb693db36fdff0))

* fix: allow _prequential_evaluation_fast to store y and predictions ([`9675615`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9675615b57b5da8c182d3857e2ca4073695ef08c))

* fix: correcting author name in ref ([`991f916`](https://github.com/adaptive-machine-learning/CapyMOA/commit/991f9161e9bd356b32b321a4c8a4729cd21bd294))

* fix(Anomaly-Detection): change expect output of example and test code for Half-Space Trees ([`5c0832a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5c0832a83a6afff0ddcee5b77a44d837ad919854))

* fix: updating HalfSpaceTrees example usage ([`fa1e236`](https://github.com/adaptive-machine-learning/CapyMOA/commit/fa1e236b38616a3656013d6f0477f1abf1f00d10))

* fix: updating moa jar 24-05-20 ([`3327f6b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3327f6b0376d995e88c1a38391ce3a58bdba5ba1))

* fix(Anomaly-Detection): change expect output for Half-Space Trees for unit testing ([`6b607c3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6b607c3da2c9a69d324d40b4ec7638f09bd90130))

* fix(Anomaly-Detection): change expect output for the code example of Half-Space Trees ([`2f7225a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2f7225aa7bea8d95da09b21a7e40c9bbef618ebb))

* fix: test 00 notebook ([`771d784`](https://github.com/adaptive-machine-learning/CapyMOA/commit/771d7848e03c75fbbb4a08eec82b4eb512a971a3))

* fix: updating ignored notebooks ([`47e174a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/47e174a2cc4fbf1f91a36b31fb1317bd3f7e3d04))

* fix: remove useless code in PI evaluator ([`ea500af`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ea500aff315c20a3f17b81d5ffa2b81026edb46e))

* fix: fine-tune transparency for prediction interval plotting ([`62c19a5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/62c19a51479fbe42f6e766cf0d2ea872a713bbe8))

* fix: add set_context in base and remove it for samknn ([`4baad0f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4baad0f959ace49b761b1df18249fb8f6fd9d766))

* fix: update importing ([`50f55ce`](https://github.com/adaptive-machine-learning/CapyMOA/commit/50f55ce0d24af33eec5b1977d7fddeb83c6b259e))

* fix: update pyproject.toml to include seaborn ([`f249056`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f249056fa490b25748746fe087431f4ff449b464))

* fix: using build_cli_str_from_mapping_and_locals for hyper generators ([`8bb4189`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8bb41891bc47e43c382e3e650e24efe1632a1531))

* fix: update again ([`7f31925`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7f31925681a7a8ee29f58f554da20fcdedc5166c))

* fix: update hyperplane generator ([`7279abc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7279abc31d45092cbeb409f3dafa38959484bfd7))

* fix: fixing a small bug for hypergenerator ([`0370553`](https://github.com/adaptive-machine-learning/CapyMOA/commit/037055378a59fe7dcc3c0b0b3fcda9b781e0373c))

* fix: use an actual moa.jar ([`b4a838a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b4a838ab8b5d34996ff1ffd38adbea0d808f609d))

* fix: update hash moa.jar (again) ([`0b7da99`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0b7da9952a8f4a40044e505d39c27b50123ecb97))

* fix: update hash value for moa.jar ([`063e1e4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/063e1e4e73e3d0fa1a439e8ea77a53ad0fdd9e87))

* fix: update the moa.jar ([`b84086f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b84086f632ddcff00ed1b263ba95d0a5375360f7))

### Refactor

* refactor(datasets): add Bike dataset into _datasets.py ([`d755e6d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d755e6dbc5e855672abf8473e560d0a202c88fd7))


## v0.2.1 (2024-05-06)

### Chore

* chore(version): increment version to 0.2.1 ([`7953523`](https://github.com/adaptive-machine-learning/CapyMOA/commit/79535231f5d2e13a9c4b75336cfe0501243eb3d1))

### Documentation

* docs: add blurb to plot and improve moajar update steps ([`cd6ca55`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cd6ca554b15087149f20607e1dc1614b4d5487bf))

* docs: use wall time in benchmark plot ([`31bb320`](https://github.com/adaptive-machine-learning/CapyMOA/commit/31bb32006856c94149f9daf5d3bb53611cbe5f5f))

### Fix

* fix: tweak docs and add links between sites ([`5eca95d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5eca95dfe8260f7a2ff337ed73f9fca68d390d0c))


## v0.2.0 (2024-05-04)

### Chore

* chore(version): increment version to 0.2.0 ([`7fc2966`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7fc29666e4a6dbf9f3ef8483c793e1b8047fe0a2))

### Documentation

* docs: fix readme for pypi ([`9c7af27`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9c7af27992b72d744264a28ee30bda45f932f46c))

### Feature

* feat: add streaming random patches classifiers ([`5d3b877`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5d3b877d9d5dbae1b38d9b77e581985dd44c050a))

### Fix

* fix: updating tutorial 6

Fixing issues with tensorboard ([`95a4d54`](https://github.com/adaptive-machine-learning/CapyMOA/commit/95a4d5499b19825adec7db3a6542299703c4bd25))

* fix: updating notebooks and more

Updated all the tutorial notebooks.
AdaptiveRandomForest -&gt; AdaptiveRandomForestClassifier.
Removed some outdated files (like accessing_sample_data.txt).
Removed outdated notebooks. ([`d1aef09`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d1aef0964181b099f5a90393be4bc81338a41db3))

### Unknown

* Create CNAME ([`44ea759`](https://github.com/adaptive-machine-learning/CapyMOA/commit/44ea7593649ff4927c047196b12f619746d43d63))

* Delete CNAME ([`049621f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/049621ffcb0d83000bbb90b84180fb48c29276bc))

* Create CNAME ([`7e0c5d5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7e0c5d5d83ed9d7aef9d10cd35d42f5c57d35264))


## v0.1.1 (2024-05-03)

### Chore

* chore(version): increment version to 0.1.1 ([`1d1cd9f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1d1cd9f20e88e57a7b554a320f1aa0eaaf949b14))

### Fix

* fix: update pyproject.toml for initial release ([`cd7279c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cd7279c7567c51a16c17a33842e53f3dd14217d4))


## v0.1.0 (2024-05-03)

### Build

* build: remove river from default dependencies ([`66a3a21`](https://github.com/adaptive-machine-learning/CapyMOA/commit/66a3a21c7e2e564decada5b7028abbddb0d1ee5a))

### Chore

* chore(version): increment version to 0.1.0 ([`1a983a2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1a983a239e7a6092e4bf9bd224687f65c0e90988))

### Ci

* ci: publish to pypi ([`ec1256c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ec1256c4f25e751c8a47bbc8246290f2131dd17a))

* ci: add gh-pages ([`6efd58c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6efd58c47c844b1047b8e0d52c80fa8580508546))

* ci: doc error ([`7c1507f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7c1507fbdcabc58aebb3f70eab900382fd60b4d9))

* ci: doc error ([`678e0a2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/678e0a213af5366f49a4dee60a45df4aec753e56))

* ci: doc error ([`3dda36c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3dda36c76a351c130d0bec44a914cfdf63f73015))

* ci: Fix doc error ([`25b34df`](https://github.com/adaptive-machine-learning/CapyMOA/commit/25b34df62182a3a835de501f60af6476e7113064))

* ci: rename SGBT to StreamingGradientBoostedTrees ([`82604a5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/82604a5d4b916a7fc3ec5d7a59b1f88ff42826ce))

* ci: change python version of github actions to 3.9 ([`1a2ca36`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1a2ca3622b6e3f8f1e7458fa460d2987ab2a38de))

* ci: only upload docs on push ([`f1f2416`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f1f2416062efb92995b201e1d645631c1d77e8b7))

* ci: add macos back to `all_targets.yml` ([`abe46f2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/abe46f278546b2327b335e7db1e96e05a4adbf90))

* ci: fix an off by one error in version numbers ([`9a8dbf1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9a8dbf1207dc9b57bf04f858fddd34461e06ece3))

### Documentation

* docs: update README ([`01c4a9f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/01c4a9f0dab4714ef194795913617d40695bd105))

* docs: update landing pages ([`ee5eee4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ee5eee4a1af6fd0a32e12376a156311abd1d2118))

* docs: fix spelling and style mistakes ([`95c72fb`](https://github.com/adaptive-machine-learning/CapyMOA/commit/95c72fb1503e291251d0252960a33f82b9ec2686))

* docs(SKClassifier): add docs, doctest, typehints, and minor refactor ([`d85b708`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d85b7081b1df0d43188fe5c0006c438a1a48b0a2))

### Feature

* feat: add missing datasets and document datasets ([`3ac973f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3ac973f7c88c3ca8cff2c09f6d15c91f597cdae4))

* feat: add online smooth boost classifiers ([`478bd93`](https://github.com/adaptive-machine-learning/CapyMOA/commit/478bd9337d453a69ce5d569907fd142e982aeb15))

* feat: add nochange and majority class classifiers ([`0c822c1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0c822c1ad3bfe7c9a9c71a9c107c710a8b0971d4))

* feat: add OzaBoost ([`a08fd1b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a08fd1b467169a932621a9a328315ff05cbece1e))

* feat: improve ``capymoa`` environment configuration ([`ee96275`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ee96275915d430cb99a2117a4d865b20fe3eafb9))

* feat: add SGBT ([`80f7007`](https://github.com/adaptive-machine-learning/CapyMOA/commit/80f7007786d76d2b82eade82a3db2f14d34d6102))

* feat(EFDT): leaf_prediction as str

Users can still use leaf_prediction as an integer (0, 1 or 2), but it can also be used as a string:
&#34;MajorityClass&#34;: 0, &#34;NaiveBayes&#34;: 1, &#34;NaiveBayesAdaptive&#34;: 2 ([`6454179`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6454179940f53557dcf2b0e859bf7cbe0f14601c))

* feat(regressor): add SGDRegressor using sklearn ([`2e55155`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2e551554b19f50a616f68dd04c213a17df300d1a))

* feat(regressor): add PassiveAggressiveRegressor ([`c54fe50`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c54fe50953801542df94f17a1594003dfc474c8b))

* feat(base): add SKRegressor ([`309aa48`](https://github.com/adaptive-machine-learning/CapyMOA/commit/309aa488b54c33c3a394c00eeed18b6203a4c8f4))

* feat(SGDClassifier): add SGDClassifier ([`ec00ffd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ec00ffd53b6e47edf1322e25283352eb02d136d6))

### Fix

* fix: update soknl and test ([`336766f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/336766f4ac09282136e534d79d020be3573f0e31))

* fix: fix python 3.9 syntax error and float comparison in test ([`c7b7c1b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c7b7c1be399452ab68b4f77604f01544464fc907))

* fix: several updates

Updated EFDT and HoeffdingTree to
use _leaf_prediction(...) from _utils.py
Also changed dataset._util.py to
dataset.utils.py
Finally, updated the tests, there were
some issues (like EFDT_gini was using
InformationGain). ([`1dc6234`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1dc6234ab89252594ed6243b152924ca5aacb114))

### Refactor

* refactor(PassiveAggressiveClassifier): use SKClassifier base class ([`cb3ff18`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cb3ff184efce6db7bc59173e5835c369378235c9))

### Unknown

* Merge branch &#39;pi-package&#39; ([`7251f7f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7251f7fe5dc41280a70e3fde9debac0d74eb8717))

* doc: create LICENSE ([`b944be0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b944be0c23b326c524d2880bca65bfc97617dcae))


## v0.0.1 (2024-04-29)

### Chore

* chore(version): increment version to 0.0.1 ([`8169a8e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8169a8eb7fd5aff7a86b30fce9e21c589668d65c))

### Ci

* ci: add conventional commit compliance check ([`136bad2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/136bad29b28240ac2a934fb0fe6b6751e2ab4db1))

* ci(gh-actions): build release ([`f165dcc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f165dccee0b60fada9fdaad867c4e64beb8c68e4))

### Documentation

* docs(EFDT): Updating the EFDT documentation ([`362a510`](https://github.com/adaptive-machine-learning/CapyMOA/commit/362a510dcf9ef6d9af0d8fee8f4cb68b125675fa))

### Fix

* fix(EFDT): fixing error with leaf_prediction added in the previous change ([`a0bb50d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a0bb50dfd9365d54bf0ff5688bd61ae19c907579))

### Unknown

* Added references in the documentation as well ([`5f4ba57`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5f4ba5773895d5630f8a4ea4b5caeff439148948))

* Updating the documentation for ARF, ARFReg and EFDT ([`f9587c8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f9587c8472eeced5c54bd438b2fe89811e22e24a))

* Fix doc error ([`a3e00ce`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a3e00ce500d7137bd2990aedf37e8fd059d353ea))

* File Rename ([`f33c9da`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f33c9da0b537cf9981c0ce1b8c183b54cca8b28d))

* Addressing review comments ([`958456f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/958456f46726d7051c82d77aea0550dbd979612c))

* Fix doc error ([`f7cffd1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f7cffd15e4916d4238c8e6bf6b246cb17027abc6))

* Split notebook 03_using_sklearn_pytorch.ipynb into 3:
* 03_0_using_sklearn.ipynb
* 03_1_using_pytorch.ipynb
* 03_2_preprocess_using_MOA.ipynb
* update README.md and invoke.yml ([`30dbbce`](https://github.com/adaptive-machine-learning/CapyMOA/commit/30dbbce4d641dc414c26e08f242e39721e344807))

* Revert &#34;Semver (#63)&#34; (#64)

This reverts commit 4da804f9986f5b5f8262998f72ab5fbd40fdb059. ([`9fa58a2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9fa58a21af08bebfe9fef81688e5dda8cda27464))

* Semver (#63)

* build(pyproject.toml): add commitizen

* build: use semantic release

* build: temporarily add `semvar` as release branch

* build: add `package.json` ([`4da804f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4da804f9986f5b5f8262998f72ab5fbd40fdb059))

* manual merge ([`8972c57`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8972c576817b0ea0fc0abf3399abe872f0c36eab))

* Merge pull request #61 from hmgomes/update-270424

updating to the latest moa.jar (prediction intervals included) ([`21dfa76`](https://github.com/adaptive-machine-learning/CapyMOA/commit/21dfa761ee69d250c461f8c424bf0e7c95fdb255))

* updating SOKNL test ([`59cea63`](https://github.com/adaptive-machine-learning/CapyMOA/commit/59cea63647944796466eecafe4957992471e51ba))

* updating to the latest moa.jar (prediction intervals included) ([`54aeac3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/54aeac3bec78fbe4e501324db7b6e7538ad1f767))

* Switch to dropbox for hosting (#59)

* Switch to dropbox for hosting

* Improve instructions for updating moajar ([`67ba9ad`](https://github.com/adaptive-machine-learning/CapyMOA/commit/67ba9ad4c3c13ed6c6d6bb554db27d2031b0b005))

* Add auto-download cli (#55) ([`2ae82ed`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2ae82ed28272d40fd5cb691b6da992b0b8effd6d))

* Update `contributing/docs.md` ([`8d77f2b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8d77f2b60c79cc534dc5cf4e28d316f7b8b861f4))

* rename pi_learner to prediction_interval ([`f1637c6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f1637c6027cef361b44c20819c0cc5bb4c8573d2))

* rename pi_learner to prediction_interval ([`1315c51`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1315c515400bce87c4baa4e11769e02ecf89aa98))

* uploading prediction interval package into capyMOA ([`bfba919`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bfba9190a1cc3bae39e44dcb74f9a755119a00ee))

* Merge pull request #53 from hmgomes/update-20-04-2024-extending-benchmark

Updates to benchmarking.py &amp; adding knn ([`1647b09`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1647b09d0841a31fb700289fdbe9ea54f75c3dad))

* Merge pull request #52 from hmgomes/fix_get_moa_creation_CLI_function

Fix get moa creation cli function ([`ed2a164`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ed2a16482bf4fa7b1eb10a101fd2266decef4c93))

* Merge pull request #47 from hmgomes/updates-18-04-2024

Removing CPSSDS for now ([`77aacfe`](https://github.com/adaptive-machine-learning/CapyMOA/commit/77aacfedcad1355eb8ab4287d12e9e07b4092ce7))

* more updates to benchmarking.py and the addition of knn classifier ([`3a10744`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3a10744a6655865b3016da1983b6900f46f199ad))

* fix _get_moa_creation_CLI function ([`52c0c49`](https://github.com/adaptive-machine-learning/CapyMOA/commit/52c0c49cf4137020e05c102a62a1751532402d4e))

* fix _get_moa_creation_CLI function ([`60da46e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/60da46e953bef5133a41d6f35dd78fbb256a7d0f))

* Merge branch &#39;main&#39; of https://github.com/hmgomes/CapyMOA into main ([`2c86db1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2c86db13335a95f1ddb6275f5835f57d13b5ec60))

* Merge pull request #51 from hmgomes/regression_ensemble

Regression ensemble ([`8a4c955`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8a4c95599ed12fe30ab49db81e0d4719ed064bee))

* updating comments and checking regressor interface ([`4e426a8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4e426a895c4f036c6436963a05ab6d7bf0362054))

* Merge branch &#39;main&#39; of https://github.com/hmgomes/CapyMOA into main ([`c773ea8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c773ea8a747d07bbec9f4eb7257832874d1daa6a))

* Merge pull request #49 from hmgomes/fix_benchmarking

Fix benchmarking ([`1efab4e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1efab4e15acb43609041d963c63f39b3a92bd841))

* updating regression testing v2 ([`db52871`](https://github.com/adaptive-machine-learning/CapyMOA/commit/db5287135d0aaa3950d67400a54b009b3e7859f7))

* updating regression testing ([`cc2ff61`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cc2ff619e3754217c55d074ee197b036b3998124))

* distinguish classification from regression in _get_moa_creation_CLI() in base.py ([`3296161`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3296161ee7c2eea5da46fe65fa149bd405964193))

* fix typing for arf-reg ([`de0f3b8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/de0f3b862d2df8272b415d466e541434634100a5))

* updating test file for regression ([`25171af`](https://github.com/adaptive-machine-learning/CapyMOA/commit/25171af509d82497f4df18df2b2934ed4aea8634))

* ensemble regressor amendments ([`8f50813`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8f508136f449863274f258fcb3cccf184b3af54d))

* Merge branch &#39;main&#39; of https://github.com/hmgomes/CapyMOA into main ([`841c225`](https://github.com/adaptive-machine-learning/CapyMOA/commit/841c225a5c51c544ea06590806ef8970094ee796))

* Fix benchmarks ([`3706636`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3706636dfa820ebbaddb0f9907c4a3bf1e0659af))

* Fix benchmarks ([`c1686ee`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c1686eea77b75b5a5bdfcb2019e7de52cc56cd58))

* Fix benchmarks ([`7576eba`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7576ebad001536910b04a7e76727b1e0f3de183b))

* Merge branch &#39;main&#39; of https://github.com/hmgomes/CapyMOA into updates-18-04-2024 ([`62dc8b6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/62dc8b6e3b8c6fe8fbfa6ddc7d0d7c782c81420b))

* Merge pull request #48 from hmgomes/updates-18-04-2024-notebook_issues

Updating notebooks to match new structure ([`205b23c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/205b23c069b642f83fbab77db7437a59934c427f))

* Solving issues with test_ssl_classifiers.py ([`6c3d4e6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6c3d4e6010a5ba058c973ad3785349c2e16538a6))

* Fixing test_batch.py issue (updating evaluation imports) ([`7efe451`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7efe4516afc0938707dfada94e47304f0bc1718f))

* also removing it from the tests ([`b631eef`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b631eef071a5dca727953d4fa2cabda91eb079f2))

* Updating notebooks to use new project structure. ([`7ff90bd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7ff90bd0221391bf131df1c758709bbfee984a1b))

* removing CPSSDS for now ([`b8955d8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b8955d84caf7983615fd299b609c4f9fcb882fd4))

* Merge pull request #40 from hmgomes/update-13042024

Updating notebooks, evaluation and visualization ([`8e11fb7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8e11fb7eacef146fa0cbb1b762e7e5c83f4bd00b))

* Merge branch &#39;main&#39; into update-13042024 ([`a9abb27`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a9abb2763b5098ed52ba16a74e07c4e77ae809bb))

* Merge pull request #44 from tachyonicClock/restructure

Restructure ([`151db25`](https://github.com/adaptive-machine-learning/CapyMOA/commit/151db251c523ce2984dbacb0dcc9982ff0afce81))

* Merge remote-tracking branch &#39;origin/main&#39; into main

# Conflicts:
#	src/capymoa/learner/regressor/regressors.py ([`eb36e60`](https://github.com/adaptive-machine-learning/CapyMOA/commit/eb36e6082d8c0404e70a4dc2a1f503b60036e063))

* Merge branch &#39;main&#39; into update-13042024 ([`cce9cc7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cce9cc7e790e32eca7473bd111061a8a15e45af9))

* Fix documentation build ([`2be493b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2be493bd62671db9ba8aff8b508557731fb4bd1b))

* Make learners private ([`b97db8c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b97db8c5050ccda2eb5cf1add50dc2c9bddc464b))

* Use nitpicky sphinx build ([`eafe1b1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/eafe1b13f09322b50ce5afe870b2c3bd76d38170))

* Update documentation ([`576c907`](https://github.com/adaptive-machine-learning/CapyMOA/commit/576c907992cb69e46ff58b204eb38b230514a997))

* Split into one learner per file ([`6c35df9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6c35df9d7ebb689afb2f7257e458b64722d45ac1))

* Run formatter and auto-fix lint issues ([`c2c320a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c2c320ad6ba0eb446ad92f001ab69c1a09d6f0bc))

* Flatten modules ([`308e3de`](https://github.com/adaptive-machine-learning/CapyMOA/commit/308e3de288a026a397428c9660837acbab26d25a))

* Add wrapper for naive bayes (#37)

* Add a wrapper for naive bayes


---------

Co-authored-by: Heitor Murilo Gomes &lt;heitor_murilo_gomes@yahoo.com.br&gt; ([`c291381`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c29138107fcb55702f8e892b0caf167dff6d2930))

* Fix tests (#43) ([`d8024d6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d8024d67a2c38ac3e7ad1139bbe9e70573f506da))

* add function for using wrappers as ensemble base learners and fixing parameter not working ([`3ea0ba8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3ea0ba8c027fe2cbafbbd28170a70e7edf995215))

* updating notebooks 00 and 01, and adding new visualization function. Some changes to evaluation.py as well. ([`d12958a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/d12958a13202a36143df1ececffc29ff3e6b1eab))

* Merge branch &#39;main&#39; into main ([`0038a51`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0038a51d48d458a15f87131c1702423d3f62c10a))

* Merge pull request #35 from tachyonicClock/Yibin

Add SOKNL wrapper into regressor.py ([`88317e6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/88317e65c26f1c6c0db940e722ca69825e81a03c))

* Update test_classifiers.py ([`137736f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/137736f9e3cf731b20a40f1cd24eaf2d63efa671))

* Merge branch &#39;main&#39; into Yibin ([`df21ccb`](https://github.com/adaptive-machine-learning/CapyMOA/commit/df21ccbeca5a5c3f10ae220d410a4416eb002d01))

* Merge pull request #34 from tachyonicClock/PassiveAggressiveClassifier

Add `PassiveAggressiveClassifier` ([`43b0471`](https://github.com/adaptive-machine-learning/CapyMOA/commit/43b0471fcb4db2714aa028aa8c7e5dcb68503de0))

* Update __init__.py ([`4a59905`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4a59905c117bcdbb7309be31a2e89b2619c1451d))

* Update README.md ([`49ebec5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/49ebec5f77876fca6cf4ea5f284ae604d06d0e7a))

* Inital version of CSVStream ([`0b3d35d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0b3d35d1806966340442b61dc76d9901ce30d091))

* Support `SplitCriterion` everywhere ([`4ff8318`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4ff8318d627ffd9cecd4ce2826e8232b73f1d112))

* Update `moa.jar` and add documentation to `regressors.py` ([`3ca5e5b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3ca5e5b5d72bf30e25e34c7299b00909a1e05637))

* update regressors.py ([`428494a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/428494aa845be642eb0a9dcce59a9ca0641b71f8))

* adding SOKNL wrapper in regressor.py ([`fa5f6c1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/fa5f6c174971bdb8664bf93f6488a776aaea310e))

* Add `PassiveAggressiveClassifier` ([`a4c206a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a4c206af1f8f397fbb6d9fb84aaa48407ce6674e))

* Test to raise warning or error if `moa.jar` changes (#33) ([`8295945`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8295945df3d179ef8b0253e1b2f92e7c781d9ac5))

* Merge pull request #32 from tachyonicClock/stream_api

Improve documentation in `capymoa.stream` ([`96b2eac`](https://github.com/adaptive-machine-learning/CapyMOA/commit/96b2eac28dcd983a4252f4798e5b01af79c7a75b))

* Update testing.md ([`0dd1d70`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0dd1d7064b3f390ffb229fa42abb871f7c3e9a8c))

* Improve documentation in `capymoa.stream` ([`5545153`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5545153fa8e0f8cc679286fb6977d4596a6ca012))

* Merge pull request #27 from hmgomes/efdt_wrapper

Wrappers for EFDT and HT ([`71cb419`](https://github.com/adaptive-machine-learning/CapyMOA/commit/71cb419fecea031be9ec2e95edadc7c3c9cf4d65))

* Fix failed tests on windows (#30) ([`a4f3c72`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a4f3c72175d0986e5e17abeddbc27e27c103c672))

* Add structure to docs (#29) ([`0f68d7d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0f68d7d6101acdd113047b26d95eb61f53d3c175))

* Simplify instance type (#28) ([`7e1052d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7e1052d8d7fc95e3a3df12c58f6f5191e9cfbdb2))

* Fixed accuracies in tests, moved tests for EFDT and HT to test_classifiers.py ([`8035c18`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8035c18f6405d467e3beee6e669f879eed8b759f))

* Added utility function for creating cli string in wrappers ([`0bf72d2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0bf72d2e647d450372c859fd4cc169c8686522d9))

* Setup tests for HT and EFDT ([`f33d9a1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f33d9a1a7f7264488871e8b415cb6a8be383f5d1))

* Updated docstrings for EFDT and HoeffdingTree ([`39988b9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/39988b97c4265a8f1ee015622e65f6b71618abc5))

* Updated docstrings for EFDT and HoeffdingTree ([`f32ddae`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f32ddaef49e087fa17fe685c5da7141a3f48d87d))

* Small fix in classifier __init__.py ([`ed976f0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ed976f077f0a1ae7921db1ec2e4f9ea0c69bbb04))

* Wrappers for EFDT and HT ([`7849cba`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7849cba6373391a7241f2ee86a1c62d497ccc79c))

* Merge branch &#39;main&#39; of github.com:hmgomes/CapyMOA into efdt_wrapper ([`c73573c`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c73573c6f94639255de7b48268b8355791efd17d))

* Torch instances (#25)

* Change doc theme

* Add numpy/pytorch instance types and change to index based classification

* Rebuild notebooks

* Fix dataset downloads

* Update doctests ([`4d9ee91`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4d9ee914a55ed5e7839442f4d764e9b59c6d9c9e))

* Merge pull request #19 from nuwangunasekara/main

Schema changes and PytorchStream ([`3603bc3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3603bc3eefa03f56ebe14271f4fa66ea17c0183e))

* Update all_targets.yml ([`98b26d0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/98b26d0bf88590e297ca3f0f33d83efceb19877b))

* Hide some internal functions as :
_init_moa_stream_and_create_moa_header,_add_instances_to_moa_stream, _numpy_to_ARFF
Rename Schema.get_schema to Schema.create_schema_from_values ([`79efb5d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/79efb5d8fc7321ef87c65d885bd454cf68279453))

* Re-run the notebook to fix execution_count issue. ([`814a768`](https://github.com/adaptive-machine-learning/CapyMOA/commit/814a7681927480f6fe2400a511f12d48d49fb276))

* Re-run the notebook to fix execution_count issue. ([`6d976b7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6d976b7353c3fde3dcf13fb9f7b1d17bfe404976))

* Re-run the notebook to fix execution_count issue. ([`996a73d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/996a73dfbc723d18da6d92edd1e5ebad3524d609))

* Re-run the notebook to fix execution_count issue. ([`12a5bb3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/12a5bb354e6e852ef37057a3f8244bde2fe2cdcc))

* * Fix doc generation issue for Scema.get_schema() ([`851f29d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/851f29d3133bd976058064dee6513f0067013e9f))

* * Fix doc generation issue for Scema.get_schema() ([`2848ee0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2848ee0dc9e83fdde54845d068063a1b5fc7597b))

* * Fix doc generation issue for Scema.get_schema() ([`7f22e2f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7f22e2fc8e2e0a502ef7beada688254c9e8e9fa7))

* * Not exposing
  * init_moa_stream_and_create_moa_header()
  * add_instances_to_moa_stream()
  * numpy_to_ARFF()
to the user.

* Add get_schema() static method to get a Schema from a passed in information ([`f6843ba`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f6843ba8d250147b4801505991968a23fd0f0d93))

* update regressors.py ([`8046510`](https://github.com/adaptive-machine-learning/CapyMOA/commit/80465108956a67a1308cb3592fea2b68ae9ac2f3))

* Update all_targets.yml ([`4249df0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4249df0c765629df08685641b97aabf67e6a62f6))

* Add doctest support (#24) ([`0ee1e30`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0ee1e30418864d8ff1657e4d3925697c61dbc40e))

* Add workflows for all targets (#23) ([`8bbc9bd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8bbc9bd429cf27948b7b8927a9026caabe7136f4))

* Improve documentation and add tasks.py (#22) ([`ef65225`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ef65225b47046d0004d20d1351bfe35d4b4ca80f))

* adding SOKNL wrapper in regressor.py ([`c679cae`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c679caef873a1224b74da505e5bb717dc2a49926))

* * Changes to Schema so that when creating, it always requires a moa_header
* helper functions to create a moa header and add instances to the moa stream
* NumpyStream uses these helpers
* Initial version of PytorchStream
 * Needs Pytorch installed in the system
 * environment.yml does not contain this requirement ([`3ebbbc5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3ebbbc5830cf64ff615cb965dd0a89961766cfc5))

* Instances (#18)

* Update instance representation
* Build notebooks ([`78ac745`](https://github.com/adaptive-machine-learning/CapyMOA/commit/78ac7455622c9fc98200ff97eaa7aba8e2fc9d3a))

* Add python stubs for java objects (#17) ([`390e577`](https://github.com/adaptive-machine-learning/CapyMOA/commit/390e577d4e08fb575300ac4ee28fb0fbce77a863))

* Add Sphinx documentation (#13) ([`5fe9cf3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5fe9cf3d9b9c7cc97c726f7876b194f6fc177d7e))

* Support more environments better (#12) ([`9e3bf45`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9e3bf450923f59370edd0acdb1acb3333351d3d7))

* Added Hoeffding tree and EFDT ([`905a706`](https://github.com/adaptive-machine-learning/CapyMOA/commit/905a7069dce35edb2f7350df8817ee8a8a334369))

* Notebook tests (#9)

* Fix cross-device copy issue
* Add notebooks to tests ([`14cccc1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/14cccc1e5b77bc1cf615048813d085b720aa2ba3))

* Cache dependencies in GitHub actions (#11) ([`cb8d9c1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cb8d9c1507beae6d6da59d59ff300cded892515c))

* Update SSL to support class values vs indices (#10) ([`1c8e577`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1c8e5776ffc68cdb38fd41f6808cf0a3e41e1fbe))

* significant changes to Instance, Schema and Evaluation. Update some of the notebooks to reflect the changes. ([`da38b49`](https://github.com/adaptive-machine-learning/CapyMOA/commit/da38b490952748e64ca3af4b98f4c0c5f5ec64f0))

* update to DriftStream example ([`a32f4d3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a32f4d3d39d75b781c3f6a5f06f1fca85efab5eb))

* aligning the DEMO and DriftStream_API notebooks ([`be407f8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/be407f8170479ccebd57cede7ebe0acd379cadb0))

* Merge pull request #8 from tachyonicClock/OSNN

OSNN ([`881d41b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/881d41b296666e9d6238f438a791a4115c774834))

* Add OSNN ([`efa8c49`](https://github.com/adaptive-machine-learning/CapyMOA/commit/efa8c49890fab1f510cc42e38da6f07b38438fab))

* Update Unit Tests workflow ([`77a6d04`](https://github.com/adaptive-machine-learning/CapyMOA/commit/77a6d04e4e35ea78ee4be260d8d42afc50d72699))

* Add instance type hints to learners ([`f5de951`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f5de95156dd01f2eba62574720fe24927c5813c1))

* Add multiclass tests to CPSSDS ([`dca87a6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dca87a618bfccbaaec1322f08bee0dedade8efc8))

* Add CovtypeTiny and fix bug in downloader ([`c17bf43`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c17bf43be43cba6b360004f638a8daf853c9e443))

* Merge pull request #7 from tachyonicClock/package

Create capymoa package ([`1fd291a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1fd291aab3d13fa22c26541b7bc7a4cd02d79367))

* Update downloading datasets (#6) ([`a9057ae`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a9057ae1fa8fa90ef7e7e7b806f3d6b412ad498a))

* Create capymoa package ([`dd38da3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/dd38da35a288f1f6f409d6936141b55208d500af))

* Adding the DriftStream API, some changes to evaluation and visualization as well to accommodate for drift visualization and passing meta-data related to drifts ([`0697b73`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0697b7353cdc29e492ffda3064086d2ffa03c9a7))

* Merge pull request #5 from tachyonicClock/cpssds

Fix CPSSDS&#39;s imports and tests ([`3519e4f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3519e4faecaa7e3526736f71e0e6f5dba1a6be5a))

* Fix CPSSDS&#39;s imports and tests ([`428d356`](https://github.com/adaptive-machine-learning/CapyMOA/commit/428d35615247fae40ff7ca4b778b1e68800e3561))

* updating environment_wds.yml (MOABridge -&gt; CapyMOA) ([`8c70317`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8c703170152777ae37220708a5061f3b80ebab73))

* updating environment.yml ([`ed3a79b`](https://github.com/adaptive-machine-learning/CapyMOA/commit/ed3a79b27ab56a5e829816794d3e3b23caa675ab))

* Update README.md ([`9629fd0`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9629fd05ad5ee54f343ec6d301533337e0d89e69))

* moving some files and updating requirements ([`58d56b5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/58d56b52b74a65f5991403d41c82953a9f646c65))

* Merge pull request #3 from tachyonicClock/modules

Add directory structure ([`3777ad7`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3777ad78fb9350b1f626868b2b3e08d7c3f3f95c))

* Update README.md

Change `MOABridge` to `CapyMOA` in `PYTHONPATH` and title. ([`39e0aa3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/39e0aa3cc7b943adc688736566dc102508c92236))

* Fix downlaod issues ([`b894d32`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b894d3219134bd488b8a05cb4e71462faff99c56))

* Merge pull request #2 from tachyonicClock/CPSSDS

CPPSDS ([`4f8e42d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4f8e42db6652c922618e2fbc06fb4cca86e4b658))

* Restructure project ([`10a6ab8`](https://github.com/adaptive-machine-learning/CapyMOA/commit/10a6ab81d13869fa003484ffcf07eadcc93f7168))

* Run formatter ([`4e52ffd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/4e52ffde2571b15ae7b102da3e938a18d4d95c88))

* Add batch classifier SSL ([`e087e5e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/e087e5e1b7697895cdc3c25cdc7173c788551b7b))

* Set python version ([`6a1a6df`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6a1a6df610b9760f380c39f07bcbf0e46d88d85b))

* Support for Windows

* Improved windows support by adding environment_wds.yml file and removing unix-specific `resource` package

* Added .idea folder to gitignore

* Remove unnecessary comments

---------

Co-authored-by: Marco Heyden &lt;heyden@i40pool.ipd.kit.edu&gt; ([`76f9c3d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/76f9c3d8d1bba0c42d618b34ffe995d5b3284b6b))

* Add CPSSDS algorithm ([`c976563`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c976563ff54b7b53b38248d182fc267bcbc7eac4))

* Add batch classifier SSL ([`12d53aa`](https://github.com/adaptive-machine-learning/CapyMOA/commit/12d53aa44ff3af7828985cbdd0d705f82747816f))

* Set python version ([`8478770`](https://github.com/adaptive-machine-learning/CapyMOA/commit/8478770fe5ed0c1ac4bd0183629a06c6f23a104b))

* updating the ensemble wrappers (ARF and OnlineBagging) and some adhoc updates to notebooks ([`3631229`](https://github.com/adaptive-machine-learning/CapyMOA/commit/36312297bfdb6d3639b61d8a514f37b5ab7616b3))

* removing reference to ensembles.py from Using_sklearn_pytorch.ipynb ([`2c1e1b4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/2c1e1b4c135d8dc4f05dfde08771b4f0cdf133b8))

* Example Creating_new_classifier notebook, removed ensembles.py, added classifiers.py and updated notebooks ([`b0e127a`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b0e127a7b64174e5803361644041abc8e11df0d1))

* Fix bug in setting optimizer on PyTorchClassifier ([`5bc2d13`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5bc2d1308b1df2fbfb58577403ebd61bab5e1793))

* Add PyTorchClassifier ([`b02fda6`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b02fda669f1d91b5a7a68f430c046f1edccb6cd3))

* Merge branch &#39;main&#39; of https://github.com/hmgomes/MOABridge ([`89c4471`](https://github.com/adaptive-machine-learning/CapyMOA/commit/89c4471c0fdd8734d81cfdf5b93092c8ab05bd2c))

* adding learners.py in lieu of MOALearners.py ([`7b34601`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7b34601221f321b88100c0cb353a8d3f54ee8d72))

* Update README to mention new notebooks ([`62d1c8d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/62d1c8d101fb6d369ca8896929d2e7b4484885a7))

* several updates, such as MOALearners.py -&gt; learners.py, adding SKClassifier wrapper, ... ([`9380a9d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9380a9dcd343f9541923ee51547116c4d7d8bce8))

* How to use TensorBoard with PyTorch ([`177c88e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/177c88e4b7c415ab84da2eaa0799d34bcee8c9c8))

* Example using simple Pytorch NeuralNetwork model with a MOA evaluator ([`1cc4e9e`](https://github.com/adaptive-machine-learning/CapyMOA/commit/1cc4e9efa691f750da87fd7715aaee34747492ea))

* Updates to examples and adding x() to Instances + notebook with example ([`02bb901`](https://github.com/adaptive-machine-learning/CapyMOA/commit/02bb901232d77dfc895585fb18f797cff13e4b4c))

* Improve MOABridge setup ([`0196fd2`](https://github.com/adaptive-machine-learning/CapyMOA/commit/0196fd254de1221885882da188fd7d064fab034b))

* Remove electricty from downloading as it is already in the repo
Add &#39;python download_datasets_and_moa.py&#39; command to README.md ([`a550405`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a5504054e178f1913143d0b0477779abef0b0e5f))

* Add script to download datasets and moa.jar ([`c599ebb`](https://github.com/adaptive-machine-learning/CapyMOA/commit/c599ebb3e1277fc25a57c35cef58ac0426450523))

* Support relative paths
Add script to download datasets and moa.jar ([`cbfe596`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cbfe59601f4fbb06f32e1760256b6e07f6554236))

* adding a ssl example ([`9d7f6dc`](https://github.com/adaptive-machine-learning/CapyMOA/commit/9d7f6dc914dcf03a48880cfd9cf5692c4428c912))

* updating the notebooks to include more instructions and use relative paths ([`bd48487`](https://github.com/adaptive-machine-learning/CapyMOA/commit/bd4848786085a12b29b96dfafd82ff475929f837))

* adding data (small files) and link to download the other files ([`013fd49`](https://github.com/adaptive-machine-learning/CapyMOA/commit/013fd496db305508debc7851c196fcd051cc8a11))

* adding the jar file and instructions for downloading the jar ([`67da91f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/67da91f61da154f322c4a9e9560cf99dfc5979fd))

* Small changes to stream and DEMO notebook, also changing the config.ini to use a relative path. ([`6716f1f`](https://github.com/adaptive-machine-learning/CapyMOA/commit/6716f1f866e555a9e0b7803d1366d8b5dee02dc6))

* Updating the readme. ([`94d3673`](https://github.com/adaptive-machine-learning/CapyMOA/commit/94d367357594e2572823e540f8ff7a00219b1628))

* Remove specific numpy==1.24.4 dependency ([`a0d81fd`](https://github.com/adaptive-machine-learning/CapyMOA/commit/a0d81fd76599f30bf396f1d1ea0acc849b936f2d))

* Adding environment.yml file to create MOABridge conda environment at ~/Desktop/MOABridge ([`7365b00`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7365b0027e4dfd7d8652c40a4ae014d04a723de7))

* lots of changes, should have made several commits, but it is what it is... ([`3daa192`](https://github.com/adaptive-machine-learning/CapyMOA/commit/3daa1927836e90984d757b884e10a469749c83e2))

* Updating all the notebooks with examples and solving a few bugs. ([`b17e7e1`](https://github.com/adaptive-machine-learning/CapyMOA/commit/b17e7e19fb8ba2e7c2ab8864bdee8d74f3d66574))

* adding all the current packages + some extra testing notebooks ([`5375180`](https://github.com/adaptive-machine-learning/CapyMOA/commit/5375180522bb2805d6d414a04762d50b62813d29))

* Several updates to evaluation (adding prequential and other changes) ([`61b632d`](https://github.com/adaptive-machine-learning/CapyMOA/commit/61b632d7b77ccb23ee252923d75dd9bd6eeeac85))

* Fixing the evaluators (there was an issue when using the moa header to initialize the schema), updated the notebook tests and also better organised the benchmarking, also uploading the experiments results ([`f4e80e5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f4e80e5b9c07357b798891f8414cc5cb0dce3644))

* Adding evaluation wrappers and an experimental version of an ARF wrapper ([`14c0fe3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/14c0fe3024d0a86fc704ed0ee051a8f096ce51a8))

* updating examples and readme ([`730b9e3`](https://github.com/adaptive-machine-learning/CapyMOA/commit/730b9e39cb56b14b91faebce72b72c1e8989fbb1))

* update the names of the notebooks and the MOA_API_Examples.ipynb ([`06947b4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/06947b42262c2cdce814654bba247a0ee10a8fab))

* updating the notebooks ([`f996919`](https://github.com/adaptive-machine-learning/CapyMOA/commit/f9969190527c738d967020019c7fcc3c30e95d73))

* added evaluation and jpype_prepare, updated the notebooks. ([`02123f5`](https://github.com/adaptive-machine-learning/CapyMOA/commit/02123f5ed088ad37c3f7ff2581fbf5701cd92cc7))

* updates to comparison notebook, adding examples for all the classifiers to be used in the comparison for both MOA and River ([`88fbe42`](https://github.com/adaptive-machine-learning/CapyMOA/commit/88fbe4293defdbd3badf1d4a3eeb21204646fd1c))

* adding a sample code for the test_train_loop script ([`7e397c9`](https://github.com/adaptive-machine-learning/CapyMOA/commit/7e397c9e7e414606c57edb076970513ba9f8cdd4))

* adding example and comparison notebook ([`cd09011`](https://github.com/adaptive-machine-learning/CapyMOA/commit/cd090114ae577202762f01870a488b968c04b416))

* Initial commit ([`378e1f4`](https://github.com/adaptive-machine-learning/CapyMOA/commit/378e1f446b1deaf5834c9904eabec7cda959da94))
