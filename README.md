# Topics in Machine Learning Accelerator Design

## Table of Contents
- [Reference Books](#books)
- [Reference Surveys](#surveys)
- [Overview of Machine Learning and Deep Learning Models](#ml-models) 
   - [DNN Model Inference and Training](#training-inference)
   - [Various Deep Learning Models](#various-models)
   - [Additional Readings](#ml-models-additional-readings)
 - [Introduction to DNN accelerators](#intro-dnn-accelerators)
   - [CNN/GEMM Accelerators](#cnn-gemm-accelerators) 
   - [NoCs for ML Accelerator](#nocs-accelerators) 
   - [Memory Design for ML Accelerator](#accelerator-memory)
 - [Accelerators for NLP](#nlp-accelerators)
   - [Accelerating RNNs and LSTMs](#rnn-lstm)
   - [Accelerating Transformers](#transformers)
 - [Neuromorphic Accelerators](#neuromorphic)
 - [Accelerator Design Frameworks and Hardware/Software Codesign Optimizations](#hw-sw-codesign)
   - [Accelerator Execution Cost Modeling](#execution-cost)
   - [Design Libraries and Frameworks](#design-frameworks)
   - [DNN-to-Accelerator Mapping Optimizations](#mapping-optimizations)
   - [Programming Languages and Compilers for Accelerator Generation](#design-by-pl-compilers)
   - [Accelerator Hardware/Software Design Space Exploration](#dse)
 - [Compilers for ML Accelerators and Machine Code Generation](#compilers-codegen)
   - [Compilers for ML Accelerator](#compilers)
   - [Domain-Specific Languages (DSLs) for ML Accelerator Programming](#dsls)
   - [Intermediate Representations (IRs)](#irs)
   - [Automated Cost Generation with Machine/Deep Learning](#automated-costs)
   - [ISAs and Machine Code Generation](#codegen)
 - [Accelerators for ML Training](#training-accelerators)
 - [Architectural Support for Quantization](#quantization)
   - [Model Compression Techniques](#model-compression)
   - [Quantization Techniques](#quantization-methods)
   - [Bit-Adaptive Computing](#bit-adaptive-computing)
   - [Leveraging Value Similarity](#value-similarity)
   - [Aggressive Quantization (Binary, Ternary, Logarithmic)](#binary-logarithmic)
 - [Sparse ML Accelerators](#sparse-mla)
 - [Accelerator-aware Neural-architecture Search and Accelerator/Model Codesigns](#nas-and-model-mla-codesign)
   - [Neural Architecture Search Techniques](#nas-techniques)
   - [Hardware-aware Neural Architecture Search](#hw-aware-nas)
   - [Accelerator-aware Neural Architecture Search](#accelerator-aware-nas)
 - [Accelerator Design for Multiple workloads and Multi-chip Accelerator Designs](#multiple-workloads-multiple-mlas)
   - [Resource Partitioning of Workloads on One or More Accelerators](#resource-partitioning)
   - [Accelerator for Multiple Workloads](#multi-workloads)
   - [NoCs for Chiplets](#noc-chiplets)
   - [Scalability Analysis](#mlas-scalability-analysis)
 - [ML Accelerators with Near-Data Processing and In-Memory Computing](#neardata-pim)
   - [ML Accelerators with Near-Data Processing](#neardata)
   - [In-Memory Acceleration](#pim) 
 - [Emerging Technologies](#emerging-tech)
   - [Photonics Accelerators](#photonics)
   - [Stochastic Computing Accelerators](#stochastic-computing)
 - [Accelerators for Recommendation Systems](#recommendation)
 - [Accelerators for Graph Learning](#graph-learning)
 - [Accelerators for Other AI Domains](#other-domains)
 - [Accelerator Benchmarking and Workload Characterizations](#benchmarks-characterizations) 
   - [ML Benchmarks for Accelerators](#benchmarks)
   - [Accelerator/Workload Characterizations for Various Deployment Scenarios](#characterizations)
   - [Simulators for ML Accelerators](#simulators)
 - [Runtime Optimizations](#runtime)
   - [Distributed Inference and Collaborative Cloud/Edge Execution](#cloud-edge-collaborative)
   - [Dynamic Adaptations](#dynamic-adaptation)
 - [On-device and Federated Learning on ML Accelerators](#federated-ondevice-learning)
   - [On-device Learning](#ondevice-learning)
   - [Federated Learning](#federated-learning)
 - [Industry Case Studies (Established Startups)](#industry-startups)
   - [Recent Industrial ML Hardware Avenues](#industry-hardware-startups)
   - [Recent Industrial ML Software Avenues](#industry-software-startups)
 - [Industry Case Studies](#industry)
 - [Reliability and Security of ML Accelerators](#reliability-security)
   - [Reliability of ML Accelerators](#reliability)
   - [Security of ML Accelerators](#security)

## <a name="books"></a> Reference Books

- Efficient processing of deep neural networks  [[Book](https://www.morganclaypool.com/doi/abs/10.2200/S01004ED1V01Y202004CAC050)] <br>
(Synthesis Lectures on Computer Architecture) <br>
Authors - Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer <br>
Publisher - Morgan and Claypool <br>
ISBN: 978-1-68-173831-4

- Deep Learning Systems: Algorithms, Compilers, and Processors for Large-Scale Production [[Book](https://www.morganclaypool.com/doi/10.2200/S01046ED1V01Y202009CAC053)] <br>
(Synthesis Lectures on Computer Architecture) <br>
Authors - Andres Rodriguez <br>
Publisher - Morgan and Claypool <br>
ISBN: 978-1-68-173966-3 <br>
EBook: https://deeplearningsystems.ai/

## <a name="surveys"></a> Reference Surveys

- (DNN Models and Architectures) Sze, Vivienne, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer. "Efficient processing of deep neural networks: A tutorial and survey." Proceedings of the IEEE 105, no. 12 (2017): 2295-2329. [[Paper](https://www.rle.mit.edu/eems/wp-content/uploads/2017/11/2017_pieee_dnn.pdf)]
- (Model Compression and Exploration Techniques) Deng, Lei, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. "Model compression and hardware acceleration for neural networks: A comprehensive survey." Proceedings of the IEEE 108, no. 4 (2020): 485-532. [[Paper](https://ieeexplore.ieee.org/abstract/document/9043731)]
- (Accelerators for Compact ML Models) Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://ieeexplore.ieee.org/abstract/document/9507542/)]
- (Deep Learning Compiler) Li, Mingzhen, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. "The deep learning compiler: A comprehensive survey." IEEE Transactions on Parallel and Distributed Systems 32, no. 3 (2020): 708-727. [[Paper](https://arxiv.org/abs/2002.03794)]


## <a name="ml-models"></a> Overview of Machine Learning and Deep Learning Models

### <a name="training-inference"></a> DNN Model Inference and Training
- Neural Networks and Deep Learning [[Online Book](http://neuralnetworksanddeeplearning.com/)] [[Videos](https://www.youtube.com/watch?v=Ijqkc7OLenI)]
  - Using neural nets to recognize handwritten digits [[Chapter1](http://neuralnetworksanddeeplearning.com/chap1.html)]
  - How the backpropagation algorithm works [[Chapter2](http://neuralnetworksanddeeplearning.com/chap2.html)]

### <a name="various-models"></a> Various Deep Learning Models
- Rodriguez, Andres. "Deep Learning Systems: Algorithms, Compilers, and Processors for Large-Scale Production." Synthesis Lectures on Computer Architecture 15, no. 4 (2020): 1-265. Chapter 3: Models and Applications. [[Chapter3](https://deeplearningsystems.ai/#ch03/)]

### <a name="ml-models-additional-readings"></a> Additional Readings

- Sze, Vivienne, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer. "Efficient processing of deep neural networks: A tutorial and survey." Proceedings of the IEEE 105, no. 12 (2017): 2295-2329. [[Paper](https://www.rle.mit.edu/eems/wp-content/uploads/2017/11/2017_pieee_dnn.pdf)] (Sections II-IV)
- Alom, Md Zahangir, Tarek M. Taha, Chris Yakopcic, Stefan Westberg, Paheding Sidike, Mst Shamima Nasrin, Mahmudul Hasan, Brian C. Van Essen, Abdul AS Awwal, and Vijayan K. Asari. "A state-of-the-art survey on deep learning theory and architectures." Electronics 8, no. 3 (2019): 292. [[Paper](https://www.mdpi.com/2079-9292/8/3/292/htm)]
- Pouyanfar, Samira, Saad Sadiq, Yilin Yan, Haiman Tian, Yudong Tao, Maria Presa Reyes, Mei-Ling Shyu, Shu-Ching Chen, and Sundaraja S. Iyengar. "A survey on deep learning: Algorithms, techniques, and applications." ACM Computing Surveys (CSUR) 51, no. 5 (2018): 1-36. [[Paper](https://courses.cs.duke.edu/spring20/compsci527/papers/Pouyanfar.pdf)]

## <a name="intro-dnn-accelerators"> </a> Introduction to DNN Accelerators

- Chen, Yu-Hsin, Tushar Krishna, Joel S. Emer, and Vivienne Sze. "Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks." IEEE journal of solid-state circuits 52, no. 1 (2016): 127-138. [[Paper](https://www.rle.mit.edu/eems/wp-content/uploads/2016/11/eyeriss_jssc_2017.pdf)]
- Chen, Yunji, Tao Luo, Shaoli Liu, Shijin Zhang, Liqiang He, Jia Wang, Ling Li et al. "Dadiannao: A machine-learning supercomputer." In 2014 47th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 609-622. IEEE, 2014. [[Paper](https://www.cs.virginia.edu/~smk9u/CS6501F16/p609-chen.pdf)]

The following papers are additional readings.

### <a name="cnn-gemm-accelerators"></a> CNN/GEMM Accelerators

- Jouppi, Norman P., Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates et al. "In-datacenter performance analysis of a tensor processing unit." In Proceedings of the 44th annual international symposium on computer architecture, pp. 1-12. 2017. [[Paper](https://arxiv.org/abs/1704.04760)]
- Chung, Eric, Jeremy Fowers, Kalin Ovtcharov, Michael Papamichael, Adrian Caulfield, Todd Massengill, Ming Liu et al. "Serving dnns in real time at datacenter scale with project brainwave." IEEE Micro 38, no. 2 (2018): 8-20. [[Paper](https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/wp/wp-microsoft-serving-dnns-in-real-time.pdf)]
- Kwon, Hyoukjun, Ananda Samajdar, and Tushar Krishna. "Maeri: Enabling flexible dataflow mapping over dnn accelerators via reconfigurable interconnects." ACM SIGPLAN Notices 53, no. 2 (2018): 461-475 [[Paper](https://cpn-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2018/01/maeri_asplos2018.pdf)]

### <a name="nocs-accelerators"></a> NoCs for ML Accelerator 

- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://arxiv.org/pdf/2007.00864.pdf)] (Section: VIII, NoC)
- Guirado, Robert, Akshay Jain, Sergi Abadal, and Eduard Alarcón. "Characterizing the communication requirements of GNN accelerators: A model-based approach." In 2021 IEEE International Symposium on Circuits and Systems (ISCAS), pp. 1-5. IEEE, 2021. [[Paper](https://arxiv.org/abs/2103.10515)]

### <a name="accelerator-memory"></a> Memory Design for ML Accelerator

- Pellauer, Michael, Yakun Sophia Shao, Jason Clemons, Neal Crago, Kartik Hegde, Rangharajan Venkatesan, Stephen W. Keckler, Christopher W. Fletcher, and Joel Emer. "Buffets: An efficient and composable storage idiom for explicit decoupled data orchestration." In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 137-151. 2019. [[Paper](https://people.eecs.berkeley.edu/~ysshao/assets/papers/Buffet_ASPLOS19_Final.pdf)]
- Dadu, Vidushi, Jian Weng, Sihao Liu, and Tony Nowatzki. "Towards general purpose acceleration by exploiting common data-dependence forms." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 924-939. 2019. [[Paper](https://polyarch.cs.ucla.edu/papers/micro2019-spu.pdf)]
- Azizimazreah, Arash, and Lizhong Chen. "Shortcut mining: Exploiting cross-layer shortcut reuse in dcnn accelerators." In 2019 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 94-105. IEEE, 2019. [[Paper](https://web.engr.oregonstate.edu/~chenliz/publications/2019_HPCA_Shortcut%20Mining.pdf)]
- Alwani, Manoj, Han Chen, Michael Ferdman, and Peter Milder. "Fused-layer CNN accelerators." In 2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 1-12. IEEE, 2016. [[Paper](http://www.ece.sunysb.edu/~pmilder/papers/16micro.pdf)]
- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://arxiv.org/pdf/2007.00864.pdf)] (Section: VII, Memory Design)

## <a name="nlp-accelerators"></a> Accelerators for NLP

- Wang, Hanrui, Zhekai Zhang, and Song Han. "Spatten: Efficient sparse attention architecture with cascade token and head pruning." In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pp. 97-110. IEEE, 2021. [[Paper](https://arxiv.org/abs/2012.09852)]

The following papers are additional readings.

### <a name="rnn-lstm"></a> Accelerating RNNs and LSTMs

- Han, Song, Junlong Kang, Huizi Mao, Yiming Hu, Xin Li, Yubin Li, Dongliang Xie et al. "Ese: Efficient speech recognition engine with sparse lstm on fpga." In Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 75-84. 2017. [[Paper](https://www.comp.nus.edu.sg/~hebs/course/cs6284/papers/han-fpga17.pdf)]
- Gao, Chang, Daniel Neil, Enea Ceolini, Shih-Chii Liu, and Tobi Delbruck. "DeltaRNN: A power-efficient recurrent neural network accelerator." In Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 21-30. 2018. [[Paper](https://www.researchgate.net/profile/Chang-Gao-10/publication/323375482_DeltaRNN_A_Power-efficient_Recurrent_Neural_Network_Accelerator/links/5fe6171a299bf1408843ec98/DeltaRNN-A-Power-efficient-Recurrent-Neural-Network-Accelerator.pdf)]
- Wang, Shuo, Zhe Li, Caiwen Ding, Bo Yuan, Qinru Qiu, Yanzhi Wang, and Yun Liang. "C-LSTM: Enabling efficient LSTM using structured compression techniques on FPGAs." In Proceedings of the 2018 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 11-20. 2018. [[Paper](https://www.researchgate.net/publication/323373772_C-LSTM_Enabling_Efficient_LSTM_using_Structured_Compression_Techniques_on_FPGAs)] 
- Gupta, Udit, Brandon Reagen, Lillian Pentecost, Marco Donato, Thierry Tambe, Alexander M. Rush, Gu-Yeon Wei, and David Brooks. "Masr: A modular accelerator for sparse rnns." In 2019 28th International Conference on Parallel Architectures and Compilation Techniques (PACT), pp. 1-14. IEEE, 2019. [[Paper](https://par.nsf.gov/servlets/purl/10161169)]
- Rezk, Nesma M., Madhura Purnaprajna, Tomas Nordström, and Zain Ul-Abdin. "Recurrent neural networks: An embedded computing perspective." IEEE Access 8 (2020): 57967-57996. [[Paper](https://arxiv.org/abs/1908.07062)]
- Yin, Shouyi, Peng Ouyang, Shibin Tang, Fengbin Tu, Xiudong Li, Shixuan Zheng, Tianyi Lu, Jiangyuan Gu, Leibo Liu, and Shaojun Wei. "A high energy efficient reconfigurable hybrid neural network processor for deep learning applications." IEEE Journal of Solid-State Circuits 53, no. 4 (2017): 968-982. [[Paper](https://ieeexplore.ieee.org/abstract/document/8207783)]
- Silfa, Franyell, Gem Dot, Jose-Maria Arnau, and Antonio Gonzàlez. "Neuron-level fuzzy memoization in rnns." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 782-793. 2019. [[Paper](https://dl.acm.org/doi/abs/10.1145/3352460.3358309)]
- Jang, Hanhwi, Joonsung Kim, Jae-Eon Jo, Jaewon Lee, and Jangwoo Kim. "Mnnfast: A fast and scalable system architecture for memory-augmented neural networks." In Proceedings of the 46th International Symposium on Computer Architecture, pp. 250-263. 2019. [[Paper](https://dl.acm.org/doi/abs/10.1145/3307650.3322214)]
- Shi, Runbin, Peiyan Dong, Tong Geng, Yuhao Ding, Xiaolong Ma, Hayden K-H. So, Martin Herbordt, Ang Li, and Yanzhi Wang. "Csb-rnn: A faster-than-realtime rnn acceleration framework with compressed structured blocks." In Proceedings of the 34th ACM International Conference on Supercomputing, pp. 1-12. 2020. [[Paper](https://www.eee.hku.hk/~casr/publication/csb-ics20/csb-ics20.pdf)]

### <a name="transformers"></a>Accelerating Transformers
- Ham, Tae Jun, Sung Jun Jung, Seonghak Kim, Young H. Oh, Yeonhong Park, Yoonho Song, Jung-Hun Park et al. "A^ 3: Accelerating attention mechanisms in neural networks with approximation." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 328-341. IEEE, 2020. [[Paper](https://arxiv.org/abs/2002.10941)]
- Tambe, Thierry, Coleman Hooper, Lillian Pentecost, Tianyu Jia, En-Yu Yang, Marco Donato, Victor Sanh et al. "Edgebert: Sentence-level energy optimizations for latency-aware multi-task nlp inference." In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 830-844. 2021. [[Paper](https://arxiv.org/abs/2011.14203)]
- Lu, Liqiang, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, and Yun Liang. "Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture." In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 977-991. 2021. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3466752.3480125)]

## <a name="neuromorphic"></a> Neuromorphic Accelerators

- Davies, Mike, Narayan Srinivasa, Tsung-Han Lin, Gautham Chinya, Yongqiang Cao, Sri Harsha Choday, Georgios Dimou et al. "Loihi: A neuromorphic manycore processor with on-chip learning." Ieee Micro 38, no. 1 (2018): 82-99. [[Paper](https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf)]

The following papers are additional readings.

- Furber, Steve B., Francesco Galluppi, Steve Temple, and Luis A. Plana. "The spinnaker project." Proceedings of the IEEE 102, no. 5 (2014): 652-665. [[Paper](https://www.researchgate.net/profile/Francesco-Galluppi/publication/262025847_The_SpiNNaker_project/links/0a85e53c4220225be0000000/The-SpiNNaker-project.pdf)]
- DeBole, M.V., Taba, B., Amir, A., Akopyan, F., Andreopoulos, A., Risk, W.P., Kusnitz, J., Otero, C.O., Nayak, T.K., Appuswamy, R. and Carlson, P.J., 2019. TrueNorth: Accelerating from zero to 64 million neurons in 10 years. Computer, 52(5), pp.20-29. [[Paper](https://ieeexplore.ieee.org/abstract/document/8713821)]
- Bouvier, Maxence, Alexandre Valentian, Thomas Mesquida, Francois Rummens, Marina Reyboz, Elisa Vianello, and Edith Beigne. "Spiking neural networks hardware implementations and challenges: A survey." ACM Journal on Emerging Technologies in Computing Systems (JETC) 15, no. 2 (2019): 1-35. [[Paper](https://arxiv.org/ftp/arxiv/papers/2005/2005.01467.pdf)]

## <a name="hw-sw-codesign"></a> Accelerator Design Frameworks and Hardware/Software Codesign Optimizations

### <a name="execution-cost"></a> Accelerator Execution Cost Modeling

- Kwon, Hyoukjun, Prasanth Chatarasi, Michael Pellauer, Angshuman Parashar, Vivek Sarkar, and Tushar Krishna. "Understanding reuse, performance, and hardware cost of dnn dataflow: A data-centric approach." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 754-768. 2019. [[Paper](https://arxiv.org/pdf/1805.02566v5.pdf)]

The following papers are additional readings.

- Wu, Yannan Nellie, Joel S. Emer, and Vivienne Sze. "Accelergy: An architecture-level energy estimation methodology for accelerator designs." In 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pp. 1-8. IEEE, 2019. [[Paper](http://accelergy.mit.edu/paper.pdf)]

### <a name="design-frameworks"></a> Design Libraries and Frameworks

The following papers are additional readings.

- Venkatesan, Rangharajan, Yakun Sophia Shao, Miaorong Wang, Jason Clemons, Steve Dai, Matthew Fojtik, Ben Keller et al. "Magnet: A modular accelerator generator for neural networks." In 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pp. 1-8. IEEE, 2019. [[Paper](https://research.nvidia.com/sites/default/files/pubs/2019-11_MAGNet%3A-A-Modular/magnet_paper.pdf)]
- Georganas, Evangelos, Dhiraj Kalamkar, Sasikanth Avancha, Menachem Adelman, Cristina Anderson, Alexander Breuer, Jeremy Bruestle et al. "Tensor processing primitives: a programming abstraction for efficiency and portability in deep learning workloads." In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-14. 2021. [[Paper](https://arxiv.org/abs/2104.05755)]
- Xu, Pengfei, Xiaofan Zhang, Cong Hao, Yang Zhao, Yongan Zhang, Yue Wang, Chaojian Li, Zetong Guan, Deming Chen, and Yingyan Lin. "AutoDNNchip: An automated dnn chip predictor and builder for both FPGAs and ASICs." In Proceedings of the 2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 40-50. 2020. [[Paper](https://arxiv.org/abs/2001.03535)]
- Zhang, Xiaofan, Junsong Wang, Chao Zhu, Yonghua Lin, Jinjun Xiong, Wen-mei Hwu, and Deming Chen. "DNNBuilder: An automated tool for building high-performance DNN hardware accelerators for FPGAs." In 2018 IEEE/ACM International Conference on Computer-Aided Design (ICCAD), pp. 1-8. IEEE, 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8587697)]
- Lai, Yi-Hsiang, Hongbo Rong, Size Zheng, Weihao Zhang, Xiuping Cui, Yunshan Jia, Jie Wang, Brendan Sullivan, Zhiru Zhang, Yun Liang, Youhui Zhang, Jason Cong, Nithin George, Jose Alvarez, , Christopher Hughes, Pradeep Dubey "Susy: A programming model for productive construction of high-performance systolic arrays on fpgas." In 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-9. IEEE, 2020. [[Paper](https://www.csl.cornell.edu/~zhiruz/pdfs/susy-iccad2020.pdf)]
- Minutoli, Marco, Vito Giovanni Castellana, Cheng Tan, Joseph Manzano, Vinay Amatya, Antonino Tumeo, David Brooks, and Gu-Yeon Wei. "Soda: a new synthesis infrastructure for agile hardware design of machine learning accelerators." In 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-7. IEEE, 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3400302.3415781)]

### <a name="mapping-optimizations"></a> DNN-to-Accelerator Mapping Optimizations

The following papers are additional readings.

- Dave, Shail, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, and Aviral Shrivastava. "DMazeRunner: Executing perfectly nested loops on dataflow accelerators." ACM Transactions on Embedded Computing Systems (TECS) 18, no. 5s (2019): 1-27. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3358198)]
- Yang, Xuan, Mingyu Gao, Qiaoyi Liu, Jeff Setter, Jing Pu, Ankita Nayak, Steven Bell et al. "Interstellar: Using halide's scheduling language to analyze dnn accelerators." In Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 369-383. 2020. [[Paper](https://arxiv.org/pdf/1809.04070.pdf)]
- Huang, Qijing, Aravind Kalaiah, Minwoo Kang, James Demmel, Grace Dinh, John Wawrzynek, Thomas Norell, and Yakun Sophia Shao. "CoSA: Scheduling by Constrained Optimization for Spatial Accelerators." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 554-566. IEEE, 2021. [[Paper](https://people.eecs.berkeley.edu/~ysshao/assets/papers/huang2021-isca.pdf)]
- Hegde, Kartik, Po-An Tsai, Sitao Huang, Vikas Chandra, Angshuman Parashar, and Christopher W. Fletcher. "Mind mappings: enabling efficient algorithm-accelerator mapping space search." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 943-958. 2021. [[Paper](https://www.kartikhegde.net/media/Mind_Mappings_ASPLOS2021_CR.pdf)]
- Mei, Linyan, Pouya Houshmand, Vikram Jain, Sebastian Giraldo, and Marian Verhelst. "ZigZag: Enlarging joint architecture-mapping design space exploration for DNN accelerators." IEEE Transactions on Computers 70, no. 8 (2021): 1160-1174. [[Paper](https://ieeexplore.ieee.org/abstract/document/9360462)]
- Chatarasi, Prasanth, Hyoukjun Kwon, Angshuman Parashar, Michael Pellauer, Tushar Krishna, and Vivek Sarkar. "Marvel: A Data-Centric Approach for Mapping Deep Learning Operators on Spatial Accelerators." ACM Transactions on Architecture and Code Optimization (TACO) 19, no. 1 (2021): 1-26. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3485137)]

### <a name="design-by-pl-compilers"></a> Programming Languages and Compilers for Accelerator Generation

The following papers are additional readings.

- LLVM CircT Project (Circuit IR Compilers and Tools). [[Talks](https://circt.llvm.org/talks)] [[GitHub](https://github.com/llvm/circt)]
- Nigam, Rachit, Samuel Thomas, Zhijing Li, and Adrian Sampson. "A compiler infrastructure for accelerator generators." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 804-817. 2021. [[Paper](https://www.cs.cornell.edu/~asampson/media/papers/calyx-asplos2021.pdf)]
- Schuiki, Fabian, Andreas Kurth, Tobias Grosser, and Luca Benini. "LLHD: A multi-level intermediate representation for hardware description languages." In Proceedings of the 41st ACM SIGPLAN Conference on Programming Language Design and Implementation, pp. 258-271. 2020. [[Paper](https://grosser.science/static/790644e9c7e6e8429b3d4958d0e08717/schuiki-2020-llhd-a-multi-level-intermediate-representation-for-hardware-description-languages.pdf)]
- Sharifian, Amirali, Reza Hojabr, Navid Rahimi, Sihao Liu, Apala Guha, Tony Nowatzki, and Arrvindh Shriraman. "μir-an intermediate representation for transforming and optimizing the microarchitecture of application accelerators." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 940-953. 2019. [[Paper](https://www2.cs.sfu.ca/~ashriram/papers/2019_MICRO_MUIR.pdf)]
- Weng, Jian, Sihao Liu, Vidushi Dadu, Zhengrong Wang, Preyas Shah, and Tony Nowatzki. "Dsagen: Synthesizing programmable spatial accelerators." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 268-281. IEEE, 2020. [[Paper](https://par.nsf.gov/servlets/purl/10177639)]
- Srivastava, Nitish, Hongbo Rong, Prithayan Barua, Guanyu Feng, Huanqi Cao, Zhiru Zhang, David Albonesi et al. "T2S-Tensor: Productively generating high-performance spatial hardware for dense tensor computations." In 2019 IEEE 27th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), pp. 181-189. IEEE, 2019. [[Paper](https://www.csl.cornell.edu/~zhiruz/pdfs/t2s-tensor-fccm2019.pdf)]
- Nigam, Rachit, Sachille Atapattu, Samuel Thomas, Zhijing Li, Theodore Bauer, Yuwei Ye, Apurva Koti, Adrian Sampson, and Zhiru Zhang. "Predictable accelerator design with time-sensitive affine types." In Proceedings of the 41st ACM SIGPLAN Conference on Programming Language Design and Implementation, pp. 393-407. 2020. [[Paper](https://par.nsf.gov/servlets/purl/10185002)]

### <a name="dse"></a> Accelerator Hardware/Software Design Space Exploration

- Nardi, Luigi, David Koeplinger, and Kunle Olukotun. "Practical design space exploration." In 2019 IEEE 27th International Symposium on Modeling, Analysis, and Simulation of Computer and Telecommunication Systems (MASCOTS), pp. 347-358. IEEE, 2019. [[Paper](https://arxiv.org/abs/1810.05236)]

The following papers are additional readings.

- Zhang, Dan, Safeen Huda, Ebrahim Songhori, Kartik Prabhu, Quoc Le, Anna Goldie, and Azalia Mirhoseini. "A full-stack search technique for domain optimized deep learning accelerators." In Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 27-42. 2022. [[Paper](https://arxiv.org/pdf/2105.12842)]
- Kao, Sheng-Chun, Geonhwa Jeong, and Tushar Krishna. "Confuciux: Autonomous hardware resource assignment for dnn accelerators using reinforcement learning." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 622-636. IEEE, 2020. [[Paper](https://www.microarch.org/micro53/papers/738300a622.pdf)]
- Wang, Jie, Licheng Guo, and Jason Cong. "Autosa: A polyhedral compiler for high-performance systolic arrays on fpga." In The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 93-104. 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3431920.3439292)]
- Xiao, Qingcheng, Size Zheng, Bingzhe Wu, Pengcheng Xu, Xuehai Qian, and Yun Liang. "Hasco: Towards agile hardware and software co-design for tensor computation." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 1055-1068. IEEE, 2021. [[Paper](https://arxiv.org/abs/2105.01585)]

## <a name="compilers-codegen"></a> Compilers for ML Accelerators and Machine Code Generation

### <a name="compilers"></a> Compilers for ML Accelerator

- Kjolstad, Fredrik, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. "The tensor algebra compiler." Proceedings of the ACM on Programming Languages 1, no. OOPSLA (2017): 1-29. [[Paper](http://groups.csail.mit.edu/commit/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf)]

The following papers are additional readings.

- Rodriguez, Andres. "Deep Learning Systems: Algorithms, Compilers, and Processors for Large-Scale Production." Synthesis Lectures on Computer Architecture 15, no. 4 (2020): 1-265. Chapter 8: Compiler Optimizations and Chapter 9: Frameworks and Compilers. [[Book](https://deeplearningsystems.ai/#ch08/)]
- Li, Mingzhen, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. "The deep learning compiler: A comprehensive survey." IEEE Transactions on Parallel and Distributed Systems 32, no. 3 (2020): 708-727. [[Paper](https://arxiv.org/abs/2002.03794)]
- Chen, Tianqi, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan et al. "TVM: An Automated {End-to-End} Optimizing Compiler for Deep Learning." In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), pp. 578-594. 2018. [[Paper](https://www.usenix.org/system/files/osdi18-chen.pdf)]
- Chou, Stephen, Fredrik Kjolstad, and Saman Amarasinghe. "Format abstraction for sparse tensor algebra compilers." Proceedings of the ACM on Programming Languages 2, no. OOPSLA (2018): 1-30. [[Paper](https://groups.csail.mit.edu/commit/papers/2018/chou-oopsla18-taco-formats.pdf)]
- Venkataramani, Swagath, Jungwook Choi, Vijayalakshmi Srinivasan, Wei Wang, Jintao Zhang, Marcel Schaal, Mauricio J. Serrano et al. "Deeptools: Compiler and execution runtime extensions for rapid AI accelerator." IEEE Micro 39, no. 5 (2019): 102-111. [[Paper](https://ieeexplore.ieee.org/abstract/document/8782645)]
- [TensorFlow XLA](https://www.tensorflow.org/xla)
- Rotem, Nadav, Jordan Fix, Saleem Abdulrasool, Garret Catron, Summer Deng, Roman Dzhabarov, Nick Gibson et al. "Glow: Graph lowering compiler techniques for neural networks." arXiv preprint arXiv:1805.00907 (2018). [[Paper](https://arxiv.org/pdf/1805.00907.pdf)]
- Wei, Richard, Lane Schwartz, and Vikram Adve. "DLVM: A modern compiler infrastructure for deep learning systems." arXiv preprint arXiv:1711.03016 (2017). [[Paper](https://arxiv.org/abs/1711.03016)]

### <a name="irs"></a> Intermediate Representations (IRs)

The following papers are additional readings.

- Lattner, Chris, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. "MLIR: A compiler infrastructure for the end of Moore's law." arXiv preprint arXiv:2002.11054 (2020). [[Paper](https://arxiv.org/abs/2002.11054)]
- Roesch, Jared, Steven Lyubomirsky, Marisa Kirisame, Logan Weber, Josh Pollock, Luis Vega, Ziheng Jiang, Tianqi Chen, Thierry Moreau, and Zachary Tatlock. "Relay: A high-level compiler for deep learning." arXiv preprint arXiv:1904.08368 (2019). [[Paper](https://arxiv.org/pdf/1904.08368.pdf)]
- Kotsifakou, Maria, Prakalp Srivastava, Matthew D. Sinclair, Rakesh Komuravelli, Vikram Adve, and Sarita Adve. "Hpvm: Heterogeneous parallel virtual machine." In Proceedings of the 23rd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, pp. 68-80. 2018. [[Paper](http://publish.illinois.edu/vikramadve/files/2021/03/18-PPOPP-HPVM.pdf)]

### <a name="dsls"> Domain-Specific Languages (DSLs) for ML Accelerator Programming

The following papers are additional readings.

- Koeplinger, David, Matthew Feldman, Raghu Prabhakar, Yaqi Zhang, Stefan Hadjis, Ruben Fiszel, Tian Zhao et al. "Spatial: A language and compiler for application accelerators." In Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation, pp. 296-311. 2018. [[Paper](http://csl.stanford.edu/~christos/publications/2018.spatial.pldi.pdf)]
- Gopinath, Sridhar, Nikhil Ghanathe, Vivek Seshadri, and Rahul Sharma. "Compiling KB-sized machine learning models to tiny IoT devices." In Proceedings of the 40th ACM SIGPLAN Conference on Programming Language Design and Implementation, pp. 79-95. 2019. [[Paper](https://www.microsoft.com/en-us/research/uploads/prod/2018/10/pldi19-SeeDot.pdf)]
- Lai, Yi-Hsiang, Yuze Chi, Yuwei Hu, Jie Wang, Cody Hao Yu, Yuan Zhou, Jason Cong, and Zhiru Zhang. "HeteroCL: A multi-paradigm programming infrastructure for software-defined reconfigurable computing." In Proceedings of the 2019 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, pp. 242-251. 2019. [[Paper](https://par.nsf.gov/servlets/purl/10094229)]

### <a name="codegen"></a> ISAs and Machine Code Generation

- Chen, Yunji, Huiying Lan, Zidong Du, Shaoli Liu, Jinhua Tao, Dong Han, Tao Luo et al. "An instruction set architecture for machine learning." ACM Transactions on Computer Systems (TOCS) 36, no. 3 (2019): 1-35. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3331469)]

The following papers are additional readings.

- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://ieeexplore.ieee.org/abstract/document/9507542/)] (Section: XII.D)
- Ambrosi, Joao, Aayush Ankit, Rodrigo Antunes, Sai Rahul Chalamalasetti, Soumitra Chatterjee, Izzat El Hajj, Guilherme Fachini, Paolo Faraboschi, Martin Foltin, Sitao Huang, Wen-mei Hwu et al. "Hardware-software co-design for an analog-digital accelerator for machine learning." In 2018 IEEE International Conference on Rebooting Computing (ICRC), pp. 1-13. IEEE, 2018. [[Paper](http://www.sitaohuang.com/publications/2018_icrc_analog_ml.pdf)]
- Srivastava, Prakalp, Mingu Kang, Sujan K. Gonugondla, Sungmin Lim, Jungwook Choi, Vikram Adve, Nam Sung Kim, and Naresh Shanbhag. "PROMISE: An end-to-end design of a programmable mixed-signal accelerator for machine-learning algorithms." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 43-56. IEEE, 2018. [[Paper](https://par.nsf.gov/servlets/purl/10078085)]

### <a name="automated-costs"></a> Automated Cost Generation with Machine/Deep Learning

The following papers are additional readings.
 
- Chen, Tianqi, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. "Learning to optimize tensor programs." Advances in Neural Information Processing Systems 31 (2018). [[Paper](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf)]
- Baghdadi, Riyadh, Massinissa Merouani, Mohamed-Hicham Leghettas, Kamel Abdous, Taha Arbaoui, and Karima Benatchba. "A deep learning based cost model for automatic code optimization." Proceedings of Machine Learning and Systems 3 (2021): 181-193. [[Paper](https://proceedings.mlsys.org/paper/2021/file/3def184ad8f4755ff269862ea77393dd-Paper.pdf)]
- Kaufman, Sam, Phitchaya Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, and Mike Burrows. "A learned performance model for tensor processing units." Proceedings of Machine Learning and Systems 3 (2021): 387-400. [[Paper](https://proceedings.mlsys.org/paper/2021/file/85d8ce590ad8981ca2c8286f79f59954-Paper.pdf)]
- Adams, Andrew, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li, Michaël Gharbi, Benoit Steiner et al. "Learning to optimize halide with tree search and random programs." ACM Transactions on Graphics (TOG) 38, no. 4 (2019): 1-12. [[Paper](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)]
- Sohrabizadeh, Atefeh, Yunsheng Bai, Yizhou Sun, and Jason Cong. "GNN-DSE: Automated Accelerator Optimization Aided by Graph Neural Networks." arXiv preprint arXiv:2111.08848 (2021). [[Paper](https://arxiv.org/pdf/2111.08848.pdf)]
- Mendis, Charith, Alex Renda, Saman Amarasinghe, and Michael Carbin. "Ithemal: Accurate, portable and fast basic block throughput estimation using deep neural networks." In International Conference on machine learning, pp. 4505-4515. PMLR, 2019. [[Paper](http://proceedings.mlr.press/v97/mendis19a/mendis19a.pdf)]
- Renda, Alex, Yishen Chen, Charith Mendis, and Michael Carbin. "Difftune: Optimizing cpu simulator parameters with learned differentiable surrogates." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 442-455. IEEE, 2020. [[Paper](https://www.microarch.org/micro53/papers/738300a442.pdf)]
- Abel, Andreas, and Jan Reineke. "Gray-box learning of serial compositions of mealy machines." In NASA Formal Methods Symposium, pp. 272-287. Springer, Cham, 2016. [[Paper](http://embedded.cs.uni-saarland.de/publications/GrayBoxLearningNFM2016.pdf)]
 
 ## <a name="training-accelerators"></a> Accelerators for Training
- Venkataramani, Swagath, Ashish Ranjan, Subarno Banerjee, Dipankar Das, Sasikanth Avancha, Ashok Jagannathan, Ajaya Durg et al. "Scaledeep: A scalable compute architecture for learning and evaluating deep networks." In Proceedings of the 44th Annual International Symposium on Computer Architecture, pp. 13-26. 2017. [[Paper](https://dl.acm.org/doi/abs/10.1145/3079856.3080244)]

The following papers are additional readings.
 
 - Song, Linghao, Jiachen Mao, Youwei Zhuo, Xuehai Qian, Hai Li, and Yiran Chen. "Hypar: Towards hybrid parallelism for deep learning accelerator array." In 2019 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 56-68. IEEE, 2019. [[Paper](https://par.nsf.gov/servlets/purl/10119087)]
- Rashidi, Saeed, Srinivas Sridharan, Sudarshan Srinivasan, and Tushar Krishna. "Astra-sim: Enabling sw/hw co-design exploration for distributed dl training platforms." In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 81-92. IEEE, 2020. [[Paper](https://synergy.ece.gatech.edu/files/2020/08/astrasim_ispass2020.pdf)]
- Das, Dipankar, Sasikanth Avancha, Dheevatsa Mudigere, Karthikeyan Vaidynathan, Srinivas Sridharan, Dhiraj Kalamkar, Bharat Kaul, and Pradeep Dubey. "Distributed deep learning using synchronous stochastic gradient descent." arXiv preprint arXiv:1602.06709 (2016). [[Paper](https://arxiv.org/pdf/1602.06709.pdf)]
- Mudigere, Dheevatsa, Yuchen Hao, Jianyu Huang, Zhihao Jia, Andrew Tulloch, Srinivas Sridharan, Xing Liu et al. "Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models." arXiv preprint arXiv:2104.05158 (2021). In ISCA 2022, Industry Track. [[Paper](https://arxiv.org/pdf/2104.05158.pdf)]
 
 ## <a name="quantization"></a> Architectural Support for Quantization
 
 - Sharify, Sayeh, Alberto Delmas Lascorz, Mostafa Mahmoud, Milos Nikolic, Kevin Siu, Dylan Malone Stuart, Zissis Poulos, and Andreas Moshovos. "Laconic deep learning inference acceleration." In 2019 ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA), pp. 304-317. IEEE, 2019. [[Paper](https://web.tecnico.ulisboa.pt/~joaomiguelvieira/public/PTDC/EEI-HAC/5215/2020/SS19a.pdf)]
 
The following papers are additional readings.
  
### <a name="model-compression"></a> Model Compression Techniques

- Deng, Lei, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. "Model compression and hardware acceleration for neural networks: A comprehensive survey." Proceedings of the IEEE 108, no. 4 (2020): 485-532. [[Paper](https://ieeexplore.ieee.org/abstract/document/9043731)]
- Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." arXiv preprint arXiv:1510.00149 (2015). In ICLR 2016. [[Paper](https://arxiv.org/abs/1510.00149)]

### <a name="quantization-methods"></a> Quantization Techniques

- Krishnamoorthi, Raghuraman. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv preprint arXiv:1806.08342 (2018). [[Paper](https://arxiv.org/pdf/1806.08342.pdf)]
- Blott, Michaela, Thomas B. Preußer, Nicholas J. Fraser, Giulio Gambardella, Kenneth O’brien, Yaman Umuroglu, Miriam Leeser, and Kees Vissers. "FINN-R: An end-to-end deep-learning framework for fast exploration of quantized neural networks." ACM Transactions on Reconfigurable Technology and Systems (TRETS) 11, no. 3 (2018): 1-23. [[Paper](https://dl.acm.org/doi/abs/10.1145/3242897)]
- Kalamkar, Dhiraj, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi et al. "A study of BFLOAT16 for deep learning training." arXiv preprint arXiv:1905.12322 (2019). [[Paper](https://arxiv.org/pdf/1905.12322)]

### <a name="bit-adaptive-computing"></a> Bit-Adaptive Computing
 
- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://ieeexplore.ieee.org/abstract/document/9507542/)] (Section: IX.A, Bit-Adaptive Computing Architectures)
- Albericio, Jorge, Alberto Delmás, Patrick Judd, Sayeh Sharify, Gerard O'Leary, Roman Genov, and Andreas Moshovos. "Bit-pragmatic deep neural network computing." In Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 382-394. 2017. [[Paper](https://www.eecg.utoronto.ca/~roman/professional/pubs/pdfs/micro17_deep_nn_moshovos_ieee.pdf)]
- Sharma, Hardik, Jongse Park, Naveen Suda, Liangzhen Lai, Benson Chau, Vikas Chandra, and Hadi Esmaeilzadeh. "Bit fusion: Bit-level dynamically composable architecture for accelerating deep neural network." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 764-775. IEEE, 2018. [[Paper](https://cseweb.ucsd.edu/~hadi/doc/paper/2018-isca-bitfusion.pdf)]
- Lee, Jinmook, Changhyeon Kim, Sanghoon Kang, Dongjoo Shin, Sangyeob Kim, and Hoi-Jun Yoo. "UNPU: An energy-efficient deep neural network accelerator with fully variable weight bit precision." IEEE Journal of Solid-State Circuits 54, no. 1 (2018): 173-185. [[Paper](https://ieeexplore.ieee.org/abstract/document/8481682)]
- Lee, Jinsu, Juhyoung Lee, Donghyeon Han, Jinmook Lee, Gwangtae Park, Hoi-Jun Yoo. "An Energy-Efficient Sparse Deep-Neural-Network Learning Accelerator With Fine-Grained Mixed Precision of FP8–FP16," in IEEE Solid-State Circuits Letters, vol. 2, no. 11, pp. 232-235, Nov. 2019, doi: 10.1109/LSSC.2019.2937440. [[Paper](https://ieeexplore.ieee.org/document/8813090)]

### <a name="value-similarity"></a> Leveraging Value Similarity

- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://ieeexplore.ieee.org/abstract/document/9507542/)] (Section: IX.C, Architectures leveraging value similarity)
- Riera, Marc, Jose-Maria Arnau, and Antonio González. "Computation reuse in DNNs by exploiting input similarity." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 57-68. IEEE, 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8416818)]
- Hegde, Kartik, Jiyong Yu, Rohit Agrawal, Mengjia Yan, Michael Pellauer, and Christopher Fletcher. "Ucnn: Exploiting computational reuse in deep neural networks via weight repetition." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 674-687. IEEE, 2018. [[Paper](https://www.researchgate.net/profile/Jiyong-Yu/publication/324600006_UCNN_Exploiting_Computational_Reuse_in_Deep_Neural_Networks_via_Weight_Repetition/links/5c590deba6fdccd6b5e39283/UCNN-Exploiting-Computational-Reuse-in-Deep-Neural-Networks-via-Weight-Repetition.pdf)]
- Buckler, Mark, Philip Bedoukian, Suren Jayasuriya, and Adrian Sampson. "EVA²: Exploiting temporal redundancy in live computer vision." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 533-546. IEEE, 2018. [[Paper](http://www.cs.cornell.edu/~asampson/media/papers/eva2-isca2018.pdf)]
- Zhu, Yuhao, Anand Samajdar, Matthew Mattina, and Paul Whatmough. "Euphrates: Algorithm-SoC Co-Design for Low-Power Mobile Continuous Vision." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 547-560. IEEE, 2018. [[Paper](https://www.cs.rochester.edu/horizon/pubs/isca18.pdf)]

### <a name="binary-logarithmic"></a> Aggressive Quantization (Binary, Ternary, Logarithmic)

- Lee, Edward H., Daisuke Miyashita, Elaina Chai, Boris Murmann, and S. Simon Wong. "Lognet: Energy-efficient neural networks using logarithmic computation." In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5900-5904. IEEE, 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/7953288)]
- Andri, Renzo, Lukas Cavigelli, Davide Rossi, and Luca Benini. "YodaNN: An architecture for ultralow power binary-weight CNN acceleration." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 37, no. 1 (2017): 48-60. [[Paper](https://ieee-ceda.org/sites/ieeeceda/files/2019-09/Pederson%20%231%20YodaNN.pdf)]
- Tann, Hokchhay, Soheil Hashemi, R. Iris Bahar, and Sherief Reda. "Hardware-software codesign of accurate, multiplier-free deep neural networks." In 2017 54th ACM/EDAC/IEEE Design Automation Conference (DAC), pp. 1-6. IEEE, 2017. [[Paper](https://www.researchgate.net/profile/Soheil-Hashemi/publication/316874913_Hardware-Software_Codesign_of_Accurate_Multiplier-free_Deep_Neural_Networks/links/59175566a6fdcc963e856150/Hardware-Software-Codesign-of-Accurate-Multiplier-free-Deep-Neural-Networks.pdf)]
 
## <a name="sparse-mla"></a> Sparse ML Accelerators

- Han, Song, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz, and William J. Dally. "EIE: Efficient inference engine on compressed deep neural network." ACM SIGARCH Computer Architecture News 44, no. 3 (2016): 243-254. [[Paper](https://arxiv.org/abs/1602.01528)]
- Guo, Cong, Bo Yang Hsueh, Jingwen Leng, Yuxian Qiu, Yue Guan, Zehuan Wang, Xiaoying Jia, Xipeng Li, Minyi Guo, and Yuhao Zhu. "Accelerating sparse dnn models without hardware-support via tile-wise sparsity." In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-15. IEEE, 2020. [[Paper](https://arxiv.org/pdf/2008.13006)]

The following papers are additional readings.
 
- Dave, Shail, Riyadh Baghdadi, Tony Nowatzki, Sasikanth Avancha, Aviral Shrivastava, and Baoxin Li. "Hardware acceleration of sparse and irregular tensor computations of ML models: A survey and insights." Proceedings of the IEEE 109, no. 10 (2021): 1706-1752. [[Paper](https://ieeexplore.ieee.org/abstract/document/9507542/)] (Section: V, Storage formats and Section VI: Extracting non-zeros and Section X: Load Balancing)
- Zhou, Xuda, Zidong Du, Qi Guo, Shaoli Liu, Chengsi Liu, Chao Wang, Xuehai Zhou, Ling Li, Tianshi Chen, and Yunji Chen. "Cambricon-S: Addressing irregularity in sparse neural networks through a cooperative software/hardware approach." In 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 15-28. IEEE, 2018. [[Paper](https://dl.acm.org/doi/abs/10.1109/micro.2018.00011)]
- Parashar, Angshuman, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rangharajan Venkatesan, Brucek Khailany, Joel Emer, Stephen W. Keckler, and William J. Dally. "SCNN: An accelerator for compressed-sparse convolutional neural networks." ACM SIGARCH computer architecture news 45, no. 2 (2017): 27-40. [[Paper](https://research.nvidia.com/sites/default/files/publications/ISCA_2017_SCNN.pdf)]
- Chen, Yu-Hsin, Tien-Ju Yang, Joel Emer, and Vivienne Sze. "Eyeriss v2: A flexible accelerator for emerging deep neural networks on mobile devices." IEEE Journal on Emerging and Selected Topics in Circuits and Systems 9, no. 2 (2019): 292-308. [[Paper](https://parsa.epfl.ch/course-info/cs723/papers/Eyerissv2.pdf)]
- Hegde, Kartik, Hadi Asghari-Moghaddam, Michael Pellauer, Neal Crago, Aamer Jaleel, Edgar Solomonik, Joel Emer, and Christopher W. Fletcher. "Extensor: An accelerator for sparse tensor algebra." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 319-333. 2019. [[Paper](https://people.csail.mit.edu/emer/papers/2019.10.micro.extensor.pdf)]
- Qin, Eric, Ananda Samajdar, Hyoukjun Kwon, Vineet Nadella, Sudarshan Srinivasan, Dipankar Das, Bharat Kaul, and Tushar Krishna. "Sigma: A sparse and irregular gemm accelerator with flexible interconnects for dnn training." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 58-70. IEEE, 2020. [[Paper](https://synergy.ece.gatech.edu/wp-content/uploads/sites/332/2020/01/sigma_hpca2020.pdf)]
- Krashinsky, Ronny, Olivier Giroux, Stephen Jones, Nick Stam, and Sridhar Ramaswamy. NVIDIA A100 Ampere Architecture In-Depth, 2020. [[Blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)]
- Liu, Zhi-Gang, Paul N. Whatmough, and Matthew Mattina. "Sparse systolic tensor array for efficient CNN hardware acceleration." arXiv preprint arXiv:2009.02381 (2020). [[Paper](https://arxiv.org/pdf/2009.02381.pdf)]
- Zmora, Neta, Guy Jacob, Lev Zlotnik, Bar Elharar, and Gal Novik. "Neural network distiller: A python package for dnn compression research." arXiv preprint arXiv:1910.12232 (2019). [[Paper](https://arxiv.org/pdf/1910.12232)]
- Sharify, Sayeh, Alberto Delmas Lascorz, Mostafa Mahmoud, Milos Nikolic, Kevin Siu, Dylan Malone Stuart, Zissis Poulos, and Andreas Moshovos. "Laconic deep learning inference acceleration." In 2019 ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA), pp. 304-317. IEEE, 2019. [[Paper](https://www.researchgate.net/profile/Sayeh-Sharify/publication/333796522_Laconic_deep_learning_inference_acceleration/links/5dcc17d9a6fdcc5750470dd4/Laconic-deep-learning-inference-acceleration.pdf)]
- Gondimalla, Ashish, Noah Chesnut, Mithuna Thottethodi, and T. N. Vijaykumar. "Sparten: A sparse tensor accelerator for convolutional neural networks." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 151-165. 2019. [[Paper](https://dl.acm.org/doi/abs/10.1145/3352460.3358291)]
- Zhu, Maohua, Tao Zhang, Zhenyu Gu, and Yuan Xie. "Sparse tensor core: Algorithm and hardware co-design for vector-wise sparse neural networks on modern gpus." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 359-371. 2019. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3352460.3358269)]
- Srivastava, Nitish, Hanchen Jin, Shaden Smith, Hongbo Rong, David Albonesi, and Zhiru Zhang. "Tensaurus: A versatile accelerator for mixed sparse-dense tensor computations." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 689-702. IEEE, 2020. [[Paper](https://par.nsf.gov/servlets/purl/10168118)]
- He, Xin, Subhankar Pal, Aporva Amarnath, Siying Feng, Dong-Hyeon Park, Austin Rovinski, Haojie Ye, Yuhan Chen, Ronald Dreslinski, and Trevor Mudge. "Sparse-TPU: Adapting systolic arrays for sparse matrices." In Proceedings of the 34th ACM International Conference on Supercomputing, pp. 1-12. 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3392717.3392751)]
- Mahmoud, Mostafa, Isak Edo, Ali Hadi Zadeh, Omar Mohamed Awad, Gennady Pekhimenko, Jorge Albericio, and Andreas Moshovos. "Tensordash: Exploiting sparsity to accelerate deep neural network training." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 781-795. IEEE, 2020. [[Paper](https://www.microarch.org/micro53/papers/738300a781.pdf)]
- Yang, Dingqing, Amin Ghasemazar, Xiaowei Ren, Maximilian Golub, Guy Lemieux, and Mieszko Lis. "Procrustes: a dataflow and accelerator for sparse deep neural network training." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 711-724. IEEE, 2020. [[Paper](https://arxiv.org/pdf/2009.10976.pdf)]

## <a name="nas-and-model-mla-codesign"></a> Accelerator-aware Neural-architecture Search and Accelerator/Model Codesigns
 
- Jiang, Weiwen, Lei Yang, Sakyasingha Dasgupta, Jingtong Hu, and Yiyu Shi. "Standing on the shoulders of giants: Hardware and neural architecture co-search with hot start." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 39, no. 11 (2020): 4154-4165. [[Paper](https://par.nsf.gov/servlets/purl/10244142)]
- Banbury, Colby, Chuteng Zhou, Igor Fedorov, Ramon Matas, Urmish Thakker, Dibakar Gope, Vijay Janapa Reddi, Matthew Mattina, and Paul Whatmough. "Micronets: Neural network architectures for deploying tinyml applications on commodity microcontrollers." Proceedings of Machine Learning and Systems 3 (2021): 517-532. [[Paper](https://proceedings.mlsys.org/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper.pdf)]

The following papers are additional readings.
 
### <a name="nas-techniques"></a> Neural Architecture Search Techniques

- Tan, Mingxing, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V. Le. "Mnasnet: Platform-aware neural architecture search for mobile." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2820-2828. 2019. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf)]
- Wu, Bichen, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, and Kurt Keutzer. "Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10734-10742. 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.pdf)]
- Cai, Han, Ligeng Zhu, and Song Han. "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware." In International Conference on Learning Representations. 2018. [[Paper](https://arxiv.org/pdf/1812.00332)]
- Wang, Tianzhe, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Hanrui Wang, Yujun Lin, and Song Han. "Apq: Joint search for network architecture, pruning and quantization policy." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2078-2087. 2020. [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.pdf)]

### <a name="hw-aware-nas"></a> Hardware-aware Neural Architecture Search

- Banbury, Colby, Chuteng Zhou, Igor Fedorov, Ramon Matas, Urmish Thakker, Dibakar Gope, Vijay Janapa Reddi, Matthew Mattina, and Paul Whatmough. "Micronets: Neural network architectures for deploying tinyml applications on commodity microcontrollers." Proceedings of Machine Learning and Systems 3 (2021): 517-532. [[Paper](https://proceedings.mlsys.org/paper/2021/file/a3c65c2974270fd093ee8a9bf8ae7d0b-Paper.pdf)]
- Benmeziane, Hadjer, Kaoutar El Maghraoui, Hamza Ouarnoughi, Smail Niar, Martin Wistuba, and Naigang Wang. "A comprehensive survey on hardware-aware neural architecture search." arXiv preprint arXiv:2101.09336 (2021). [[Paper](https://arxiv.org/pdf/2101.09336.pdf)]

### <a name="accelerator-aware-nas"></a> Accelerator-aware Neural Architecture Search

- Jiang, Weiwen, Xinyi Zhang, Edwin H-M. Sha, Lei Yang, Qingfeng Zhuge, Yiyu Shi, and Jingtong Hu. "Accuracy vs. efficiency: Achieving both through fpga-implementation aware neural architecture search." In Proceedings of the 56th Annual Design Automation Conference 2019, pp. 1-6. 2019. [[Paper](https://arxiv.org/abs/1901.11211)]
- Li, Yuhong, Cong Hao, Xiaofan Zhang, Xinheng Liu, Yao Chen, Jinjun Xiong, Wen-mei Hwu, and Deming Chen. "Edd: Efficient differentiable dnn architecture and implementation co-search for embedded ai solutions." In 2020 57th ACM/IEEE Design Automation Conference (DAC), pp. 1-6. IEEE, 2020. [[Paper](https://www.researchgate.net/profile/Cong-Hao/publication/341202517_EDD_Efficient_Differentiable_DNN_Architecture_and_Implementation_Co-search_for_Embedded_AI_Solutions/links/5fb2e72ca6fdcc9ae05afbce/EDD-Efficient-Differentiable-DNN-Architecture-and-Implementation-Co-search-for-Embedded-AI-Solutions.pdf)]
- Jiang, Weiwen, Lei Yang, Sakyasingha Dasgupta, Jingtong Hu, and Yiyu Shi. "Standing on the shoulders of giants: Hardware and neural architecture co-search with hot start." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 39, no. 11 (2020): 4154-4165. [[Paper](https://par.nsf.gov/servlets/purl/10244142)]
- Zhou, Yanqi, Xuanyi Dong, Berkin Akin, Mingxing Tan, Daiyi Peng, Tianjian Meng, Amir Yazdanbakhsh, Da Huang, Ravi Narayanaswami, and James Laudon. "Rethinking co-design of neural architectures and hardware accelerators." arXiv preprint arXiv:2102.08619 (2021). [[Paper](https://arxiv.org/pdf/2102.08619.pdf)]
- Kumar, Aviral, Amir Yazdanbakhsh, Milad Hashemi, Kevin Swersky, and Sergey Levine. "Data-Driven Offline Optimization For Architecting Hardware Accelerators." arXiv preprint arXiv:2110.11346 (2021). In ICLR 2022. [[Paper](https://arxiv.org/pdf/2110.11346)]
- Lin, Yujun, Mengtian Yang, and Song Han. "NAAS: Neural accelerator architecture search." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 1051-1056. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2105.13258.pdf)]
- Choi, Kanghyun, Deokki Hong, Hojae Yoon, Joonsang Yu, Youngsok Kim, and Jinho Lee. "Dance: Differentiable accelerator/network co-exploration." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 337-342. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2009.06237.pdf)]

 
## <a name="multiple-workloads-multiple-mlas"></a> Accelerator Design for Multiple workloads and Multi-chip Accelerator Designs
 
- Shao, Yakun Sophia, Jason Clemons, Rangharajan Venkatesan, Brian Zimmer, Matthew Fojtik, Nan Jiang, Ben Keller et al. "Simba: Scaling deep-learning inference with multi-chip-module-based architecture." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 14-27. 2019. [[Paper](https://dl.acm.org/doi/abs/10.1145/3352460.3358302)]
- Kwon, Hyoukjun, Liangzhen Lai, Michael Pellauer, Tushar Krishna, Yu-Hsin Chen, and Vikas Chandra. "Heterogeneous dataflow accelerators for multi-DNN workloads." In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pp. 71-83. IEEE, 2021. [[Paper](https://arxiv.org/abs/1909.07437)]
 
The following papers are additional readings.

### <a name="resource-partitioning"></a> Resource Partitioning of Workloads on One or More Accelerators

- Shen, Yongming, Michael Ferdman, and Peter Milder. "Maximizing CNN accelerator efficiency through resource partitioning." In 2017 ACM/IEEE 44th Annual International Symposium on Computer Architecture (ISCA), pp. 535-547. IEEE, 2017. [[Paper](https://www.petermilder.com/assets/pdf/papers/17isca.pdf)]
- Song, Linghao, Fan Chen, Youwei Zhuo, Xuehai Qian, Hai Li, and Yiran Chen. "Accpar: Tensor partitioning for heterogeneous deep learning accelerators." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 342-355. IEEE, 2020. [[Paper](http://alchem.usc.edu/portal/static/download/accpar.pdf)]
- Ghodrati, Soroush, Byung Hoon Ahn, Joon Kyung Kim, Sean Kinzer, Brahmendra Reddy Yatham, Navateja Alla, Hardik Sharma et al. "Planaria: Dynamic architecture fission for spatial multi-tenant acceleration of deep neural networks." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 681-697. IEEE, 2020. [[Paper](https://www.microarch.org/micro53/papers/738300a681.pdf)]
- Choi, Yujeong, and Minsoo Rhu. "Prema: A predictive multi-task scheduling algorithm for preemptible neural processing units." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 220-233. IEEE, 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9065590)]
- Baek, Eunjin, Dongup Kwon, and Jangwoo Kim. "A multi-neural network acceleration architecture." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 940-953. IEEE, 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9138929)]

### <a name="multi-workloads"></a> Accelerator for Multiple Workloads
- Boroumand, Amirali, Saugata Ghose, Berkin Akin, Ravi Narayanaswami, Geraldo F. Oliveira, Xiaoyu Ma, Eric Shiu, and Onur Mutlu. "Google Neural Network Models for Edge Devices: Analyzing and Mitigating Machine Learning Inference Bottlenecks." In 2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT), pp. 159-172. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2109.14320)]

### <a name="noc-chiplets"></a> NoC for Chiplets
- Guirado, Robert, Hyoukjun Kwon, Sergi Abadal, Eduard Alarcón, and Tushar Krishna. "Dataflow-architecture co-design for 2.5 d dnn accelerators using wireless network-on-package." In 2021 26th Asia and South Pacific Design Automation Conference (ASP-DAC), pp. 806-812. IEEE, 2021. [[Paper](https://www.aspdac.com/aspdac2021/archive/pdf/9B-2.pdf)]
- Li, Yuan, Ahmed Louri, and Avinash Karanth. "Scaling Deep-Learning Inference with Chiplet-based Architecture and Photonic Interconnects." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 931-936. IEEE, 2021. [[Paper](https://oucsace.cs.ohio.edu/~avinashk/papers/DAC2021.pdf)]
- Zimmer, B., Venkatesan, R., Shao, Y.S., Clemons, J., Fojtik, M., Jiang, N., Keller, B., Klinefelter, A., Pinckney, N., Raina, P. and Tell, S.G., 2020. A 0.32–128 TOPS, scalable multi-chip-module-based deep neural network inference accelerator with ground-referenced signaling in 16 nm. IEEE Journal of Solid-State Circuits, 55(4), pp.920-932. [[Paper](https://people.csail.mit.edu/emer/papers/2019.09.jssc.rc-18.pdf)]

### <a name="mlas-scalability-analysis"></a> Scalability Analysis
- Samajdar, Ananda, Jan Moritz Joseph, Yuhao Zhu, Paul Whatmough, Matthew Mattina, and Tushar Krishna. "A systematic methodology for characterizing scalability of dnn accelerators using scale-sim." In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 58-68. IEEE, 2020. [[Paper](https://synergy.ece.gatech.edu/wp-content/uploads/sites/332/2020/03/scalesim_ispass2020.pdf)]

## <a name="neardata-pim"></a> ML Accelerators with Near-Data Processing and In-Memory Computing

- Gao, Mingyu, Jing Pu, Xuan Yang, Mark Horowitz, and Christos Kozyrakis. "Tetris: Scalable and efficient neural network acceleration with 3d memory." In Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 751-764. 2017. [[Paper](http://csl.stanford.edu/~christos/publications/2017.tetris.asplos.pdf)]
- Shafiee, Ali, Anirban Nag, Naveen Muralimanohar, Rajeev Balasubramonian, John Paul Strachan, Miao Hu, R. Stanley Williams, and Vivek Srikumar. "ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars." In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pp. 14-26. IEEE, 2016. [[Paper](http://www.cs.utah.edu/~rajeev/pubs/isca16-old.pdf)]

The following papers are additional readings.

### <a name="neardata"></a> ML Accelerators with Near-Data Processing

- Kwon, Youngeun, Yunjae Lee, and Minsoo Rhu. "Tensordimm: A practical near-memory processing architecture for embeddings and tensor operations in deep learning." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 740-753. 2019. [[Paper](https://www.ndsl.kaist.edu/computing-lunch/data/TensorDIMM_MICRO52_upload.pdf)]
- Eckert, Charles, Xiaowei Wang, Jingcheng Wang, Arun Subramaniyan, Ravi Iyer, Dennis Sylvester, David Blaaauw, and Reetuparna Das. "Neural cache: Bit-serial in-cache acceleration of deep neural networks." In 2018 ACM/IEEE 45Th annual international symposium on computer architecture (ISCA), pp. 383-396. IEEE, 2018. [[Paper](http://blaauw.engin.umich.edu/wp-content/uploads/sites/342/2018/10/Eckert-Naural-Cache.pdf)]
- Boroumand, Amirali, Saugata Ghose, Minesh Patel, Hasan Hassan, Brandon Lucia, Rachata Ausavarungnirun, Kevin Hsieh et al. "CoNDA: Efficient cache coherence support for near-data accelerators." In Proceedings of the 46th International Symposium on Computer Architecture, pp. 629-642. 2019. [[Paper](https://people.inf.ethz.ch/omutlu/pub/CONDA-coherence-for-near-data-accelerators_isca19.pdf)]


### <a name="pim"></a> In-Memory Acceleration

- Chi, Ping, Shuangchen Li, Cong Xu, Tao Zhang, Jishen Zhao, Yongpan Liu, Yu Wang, and Yuan Xie. "PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory." In 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA), pp. 27-39. IEEE, 2016. [[Paper](https://iscaconf.org/isca2016/wp-content/uploads/2016/07/1A-3.pdf)]
- Song, Linghao, Xuehai Qian, Hai Li, and Yiran Chen. "Pipelayer: A pipelined reram-based accelerator for deep learning." In 2017 IEEE international symposium on high performance computer architecture (HPCA), pp. 541-552. IEEE, 2017. [[Paper](http://alchem.usc.edu/portal/static/download/nn_memristor.pdf)]
- Imani, Mohsen, Saransh Gupta, Yeseong Kim, and Tajana Rosing. "Floatpim: In-memory acceleration of deep neural network training with high precision." In 2019 ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA), pp. 802-815. IEEE, 2019. [[Paper](https://par.nsf.gov/servlets/purl/10108128)]
- Koppula, Skanda, Lois Orosa, A. Giray Yağlıkçı, Roknoddin Azizi, Taha Shahroodi, Konstantinos Kanellopoulos, and Onur Mutlu. "EDEN: Enabling energy-efficient, high-performance deep neural network inference using approximate DRAM." In Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, pp. 166-181. 2019. [[Paper](https://people.inf.ethz.ch/omutlu/pub/EDEN-efficient-DNN-inference-with-approximate-memory_micro19.pdf)]
- Deng, Quan, Lei Jiang, Youtao Zhang, Minxuan Zhang, and Jun Yang. "Dracc: a dram based accelerator for accurate cnn inference." In Proceedings of the 55th annual design automation conference, pp. 1-6. 2018. [[Paper](https://par.nsf.gov/servlets/purl/10087543)]


## <a name="emerging-tech"></a> Emerging Technologies

The following papers are additional readings.

### <a name="photonics"></a> Photonics Accelerators

- Sunny, Febin, Asif Mirza, Mahdi Nikdast, and Sudeep Pasricha. "CrossLight: A cross-layer optimized silicon photonic neural network accelerator." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 1069-1074. IEEE, 2021. [[Paper](https://par.nsf.gov/servlets/purl/10230933)]
- Sunny, Febin P., Ebadollah Taheri, Mahdi Nikdast, and Sudeep Pasricha. "A survey on silicon photonics for deep learning." ACM Journal of Emerging Technologies in Computing System 17, no. 4 (2021): 1-57. [[Paper](https://par.nsf.gov/servlets/purl/10230939)]

### <a name="stochastic-computing"></a> Stochastic Computing Accelerators

- Li, Shuangchen, Alvin Oliver Glova, Xing Hu, Peng Gu, Dimin Niu, Krishna T. Malladi, Hongzhong Zheng, Bob Brennan, and Yuan Xie. "Scope: A stochastic computing engine for dram-based in-situ accelerator." In 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 696-709. IEEE, 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8574579/)]
- Sim, Hyeonuk, Dong Nguyen, Jongeun Lee, and Kiyoung Choi. "Scalable stochastic-computing accelerator for convolutional neural networks." In 2017 22nd Asia and South Pacific Design Automation Conference (ASP-DAC), pp. 696-701. IEEE, 2017. [[Paper](https://www.aspdac.com/aspdac2017/archive/pdf/8B-4_add_file.pdf)]
- Sim, Hyeonuk, and Jongeun Lee. "A new stochastic computing multiplier with application to deep convolutional neural networks." In Proceedings of the 54th Annual Design Automation Conference 2017, pp. 1-6. 2017. [[Paper](https://dl.acm.org/doi/abs/10.1145/3061639.3062290)]
- Hojabr, Reza, Kamyar Givaki, SM Reza Tayaranian, Parsa Esfahanian, Ahmad Khonsari, Dara Rahmati, and M. Hassan Najafi. "Skippynn: An embedded stochastic-computing accelerator for convolutional neural networks." In 2019 56th ACM/IEEE Design Automation Conference (DAC), pp. 1-6. IEEE, 2019. [[Paper](https://www.researchgate.net/profile/M-Hassan-Najafi/publication/333336489_SkippyNN_An_Embedded_Stochastic-Computing_Accelerator_for_Convolutional_Neural_Networks/links/5d2d6898a6fdcc2462e30cae/SkippyNN-An-Embedded-Stochastic-Computing-Accelerator-for-Convolutional-Neural-Networks.pdf)]

## <a name="recommendation"></a> Accelerators for Recommendation Systems
- Ke, Liu, Udit Gupta, Benjamin Youngjae Cho, David Brooks, Vikas Chandra, Utku Diril, Amin Firoozshahian et al. "Recnmp: Accelerating personalized recommendation with near-memory processing." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 790-803. IEEE, 2020. [[Paper](https://hsienhsinlee.github.io/MARS/pub/isca2020-1.pdf)]

The following papers are additional readings.

- Gupta, Udit, Carole-Jean Wu, Xiaodong Wang, Maxim Naumov, Brandon Reagen, David Brooks, Bradford Cottel et al. "The architectural implications of facebook's dnn-based personalized recommendation." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 488-501. IEEE, 2020. [[Paper](https://parsa.epfl.ch/course-info/cs723/papers/hpca-2020-facebook.pdf)]
- Hwang, Ranggi, Taehun Kim, Youngeun Kwon, and Minsoo Rhu. "Centaur: A chiplet-based, hybrid sparse-dense accelerator for personalized recommendations." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 968-981. IEEE, 2020. [[Paper](https://asset-pdf.scinapse.io/prod/3042495273/3042495273.pdf)]
- Gupta, Udit, Samuel Hsia, Jeff Zhang, Mark Wilkening, Javin Pombra, Hsien-Hsin Sean Lee, Gu-Yeon Wei, Carole-Jean Wu, and David Brooks. "RecPipe: Co-designing Models and Hardware to Jointly Optimize Recommendation Quality and Performance." In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 870-884. 2021. [[Paper](https://hsienhsinlee.github.io/MARS/pub/micro2021.pdf)]
- Zhu, Yu, Zhenhao He, Wenqi Jiang, Kai Zeng, Jingren Zhou, and Gustavo Alonso. "Distributed recommendation inference on fpga clusters." In 2021 31st International Conference on Field-Programmable Logic and Applications (FPL), pp. 279-285. IEEE, 2021. [[Paper](https://kai-zeng.github.io/papers/FPL.pdf)]


## <a name="graph-learning></a> Accelerators for Graph Learning

- Garg, Raveesh, Eric Qin, Francisco Muñoz-Martínez, Robert Guirado, Akshay Jain, Sergi Abadal, José L. Abellán et al. "Understanding the Design Space of Sparse/Dense Multiphase Dataflows for Mapping Graph Neural Networks on Spatial Accelerators." arXiv preprint arXiv:2103.07977 (2021). In IPDPS 2022. [[Paper](https://arxiv.org/pdf/2103.07977.pdf)]

The following papers are additional readings.

- Yan, Mingyu, Lei Deng, Xing Hu, Ling Liang, Yujing Feng, Xiaochun Ye, Zhimin Zhang, Dongrui Fan, and Yuan Xie. "Hygcn: A gcn accelerator with hybrid architecture." In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), pp. 15-29. IEEE, 2020. [[Paper](https://par.nsf.gov/servlets/purl/10188415)]
- Geng, Tong, Ang Li, Runbin Shi, Chunshu Wu, Tianqi Wang, Yanfei Li, Pouya Haghi et al. "AWB-GCN: A graph convolutional network accelerator with runtime workload rebalancing." In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 922-936. IEEE, 2020. [[Paper](https://par.nsf.gov/servlets/purl/10195307)]
- Liang, Shengwen, Ying Wang, Cheng Liu, Lei He, L. I. Huawei, Dawen Xu, and Xiaowei Li. "Engn: A high-throughput and energy-efficient accelerator for large graph neural networks." IEEE Transactions on Computers 70, no. 9 (2020): 1511-1525. [[Paper](https://ieeexplore.ieee.org/abstract/document/9161360)]
- Kiningham, Kevin, Christopher Re, and Philip Levis. "GRIP: A graph neural network accelerator architecture." arXiv preprint arXiv:2007.13828 (2020). [[Paper](https://arxiv.org/pdf/2007.13828)]
- Li, Jiajun, Ahmed Louri, Avinash Karanth, and Razvan Bunescu. "GCNAX: A flexible and energy-efficient accelerator for graph convolutional neural networks." In 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pp. 775-788. IEEE, 2021. [[Paper](https://oucsace.cs.ohio.edu/~avinashk/papers/HPCA_GCNAX.pdf)]
- Stevens, Jacob R., Dipankar Das, Sasikanth Avancha, Bharat Kaul, and Anand Raghunathan. "Gnnerator: A hardware/software framework for accelerating graph neural networks." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 955-960. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2103.10836.pdf)]
- Liang, Shengwen, Cheng Liu, Ying Wang, Huawei Li, and Xiaowei Li. "Deepburning-gl: an automated framework for generating graph neural network accelerators." In 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-9. IEEE, 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9256539)]
- Zhang, Yongan, Haoran You, Yonggan Fu, Tong Geng, Ang Li, and Yingyan Lin. "G-CoS: GNN-Accelerator Co-Search Towards Both Better Accuracy and Efficiency." In 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD), pp. 1-9. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2109.08983.pdf)]
- Song, Xinkai, Tian Zhi, Zhe Fan, Zhenxing Zhang, Xi Zeng, Wei Li, Xing Hu, Zidong Du, Qi Guo, and Yunji Chen. "Cambricon-G: A Polyvalent Energy-Efficient Accelerator for Dynamic Graph Neural Networks." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 41, no. 1 (2021): 116-128. [[Paper](https://ieeexplore.ieee.org/abstract/document/9326339)]


## <a name="other-domains></a> Accelerators for Other AI Domains

The following papers are additional readings.

- Yazdanbakhsh, Amir, Kambiz Samadi, Nam Sung Kim, and Hadi Esmaeilzadeh. "Ganax: A unified mimd-simd acceleration for generative adversarial networks." In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), pp. 650-661. IEEE, 2018. [[Paper](https://par.nsf.gov/servlets/purl/10077001)]
- Hegde, Kartik, Rohit Agrawal, Yulun Yao, and Christopher W. Fletcher. "Morph: Flexible acceleration for 3d cnn-based video understanding." In 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 933-946. IEEE, 2018. [[Paper](https://www.kartikhegde.net/media/Morph_final_CR.pdf)]


## <a name="benchmarks-characterizations"></a> Accelerator Benchmarking and Workload Characterizations

### <a name="benchmarks"></a> ML Benchmarks for Accelerators

- Reddi, Vijay Janapa, Christine Cheng, David Kanter, Peter Mattson, Guenther Schmuelling, Carole-Jean Wu, Brian Anderson et al. "Mlperf inference benchmark." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 446-459. IEEE, 2020. [[Paper](https://arxiv.org/pdf/1911.02549.pdf)]

The following papers are additional readings.

- Mattson, Peter, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang et al. "Mlperf training benchmark." Proceedings of Machine Learning and Systems 2 (2020): 336-349. [[Paper](https://proceedings.mlsys.org/paper/2020/file/02522a2b2726fb0a03bb19f2d8d9524d-Paper.pdf)]
- Banbury, Colby, Vijay Janapa Reddi, Peter Torelli, Jeremy Holleman, Nat Jeffries, Csaba Kiraly, Pietro Montino et al. "Mlperf tiny benchmark." arXiv preprint arXiv:2106.07597 (2021). [[Paper](https://arxiv.org/pdf/2106.07597.pdf)]


### <a name="characterizations"></a> Accelerator/Workload Characterizations for Various Deployment Scenarios

- Park, Jongsoo, Maxim Naumov, Protonu Basu, Summer Deng, Aravind Kalaiah, Daya Khudia, James Law et al. "Deep learning inference in facebook data centers: Characterization, performance optimizations and hardware implications." arXiv preprint arXiv:1811.09886 (2018). [[Paper](https://arxiv.org/pdf/1811.09886)]

The following papers are additional readings.

- Wang, Yu Emma, Gu-Yeon Wei, and David Brooks. "Benchmarking tpu, gpu, and cpu platforms for deep learning." arXiv preprint arXiv:1907.10701 (2019). [[Paper](https://arxiv.org/abs/1907.10701)]
- Hadidi, Ramyad, Jiashen Cao, Yilun Xie, Bahar Asgari, Tushar Krishna, and Hyesoon Kim. "Characterizing the deployment of deep neural networks on commercial edge devices." In 2019 IEEE International Symposium on Workload Characterization (IISWC), pp. 35-48. IEEE, 2019. [[Paper](https://par.nsf.gov/servlets/purl/10195735)]
- Ignatov, Andrey, Radu Timofte, Andrei Kulik, Seungsoo Yang, Ke Wang, Felix Baum, Max Wu, Lirong Xu, and Luc Van Gool. "Ai benchmark: All about deep learning on smartphones in 2019." In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pp. 3617-3635. IEEE, 2019. [[Paper](http://people.ee.ethz.ch/~timofter/publications/Ignatov-ICCVW-2019a.pdf)]
- Yazdanbakhsh, Amir, Kiran Seshadri, Berkin Akin, James Laudon, and Ravi Narayanaswami. "An evaluation of edge tpu accelerators for convolutional neural networks." arXiv preprint arXiv:2102.10423 (2021). [[Paper](https://arxiv.org/pdf/2102.10423)]
- Pullini, Antonio, Davide Rossi, Igor Loi, Giuseppe Tagliavini, and Luca Benini. "Mr. Wolf: An energy-precision scalable parallel ultra low power SoC for IoT edge processing." IEEE Journal of Solid-State Circuits 54, no. 7 (2019): 1970-1981. [[Paper](https://www.pulp-platform.org/docs/publications/08715500.pdf)]
- Reuther, Albert, Peter Michaleas, Michael Jones, Vijay Gadepally, Siddharth Samsi, and Jeremy Kepner. "Survey and benchmarking of machine learning accelerators." In 2019 IEEE high performance extreme computing conference (HPEC), pp. 1-9. IEEE, 2019. [[Paper](https://arxiv.org/pdf/1908.11348.pdf)]


### <a name="simulators"></a> Simulators for ML Accelerators

The following papers are additional readings.

- Peng, Xiaochen, Shanshi Huang, Yandong Luo, Xiaoyu Sun, and Shimeng Yu. "DNN+ NeuroSim: An end-to-end benchmarking framework for compute-in-memory accelerators with versatile device technologies." In 2019 IEEE international electron devices meeting (IEDM), pp. 32-5. IEEE, 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8993491)]
- Xi, Sam, Yuan Yao, Kshitij Bhardwaj, Paul Whatmough, Gu-Yeon Wei, and David Brooks. "SMAUG: End-to-end full-stack simulation infrastructure for deep learning workloads." ACM Transactions on Architecture and Code Optimization (TACO) 17, no. 4 (2020): 1-26. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3424669)]
- Genc, Hasan, Seah Kim, Alon Amid, Ameer Haj-Ali, Vighnesh Iyer, Pranav Prakash, Jerry Zhao et al. "Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration." In 2021 58th ACM/IEEE Design Automation Conference (DAC), pp. 769-774. IEEE, 2021. [[Paper](https://arxiv.org/abs/1911.09925)]
- Muñoz-Martínez, Francisco, José L. Abellán, Manuel E. Acacio, and Tushar Krishna. "STONNE: Enabling Cycle-Level Microarchitectural Simulation for DNN Inference Accelerators." In 2021 IEEE International Symposium on Workload Characterization (IISWC), pp. 201-213. IEEE, 2021. [[Paper](https://arxiv.org/pdf/2006.07137)]
- Agostini, Nicolas Bohm, Shi Dong, Elmira Karimi, Marti Torrents Lapuerta, José Cano, José L. Abellán, and David Kaeli. "Design space exploration of accelerators and end-to-end DNN evaluation with TFLITE-SOC." In 2020 IEEE 32nd International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD), pp. 10-19. IEEE, 2020. [[Paper](http://www.dcs.gla.ac.uk/~josecr/pub/2020_sbac-pad.pdf)]

## <a name="runtime"></a> Runtime Optimizations

### <a name="cloud-edge-collaborative"></a> Distributed Inference and Collaborative Cloud/Edge Execution

- Kang, Yiping, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, and Lingjia Tang. "Neurosurgeon: Collaborative intelligence between the cloud and mobile edge." ACM SIGARCH Computer Architecture News 45, no. 1 (2017): 615-629. [[Paper](http://tnm.engin.umich.edu/wp-content/uploads/sites/353/2018/10/2017.04.neurosurgeonASPLOS.pdf)]

The following papers are additional readings.

- Teerapittayanon, Surat, Bradley McDanel, and Hsiang-Tsung Kung. "Distributed deep neural networks over the cloud, the edge and end devices." In 2017 IEEE 37th international conference on distributed computing systems (ICDCS), pp. 328-339. IEEE, 2017. [[Paper](http://www.eecs.harvard.edu/~htk/publication/2017-icdcs-teerapittayanon-mcdanel-kung.pdf)]
- Eshratifar, Amir Erfan, Mohammad Saeed Abrishami, and Massoud Pedram. "JointDNN: an efficient training and inference engine for intelligent mobile cloud computing services." IEEE Transactions on Mobile Computing 20, no. 2 (2019): 565-576. [[Paper](https://mpedram.com/~massoud/Papers/research_projects_papers/Amirerfan/Joint/jointdnn.pdf)]
- Hu, Chuang, Wei Bao, Dan Wang, and Fengming Liu. "Dynamic adaptive DNN surgery for inference acceleration on the edge." In IEEE INFOCOM 2019-IEEE Conference on Computer Communications, pp. 1423-1431. IEEE, 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8737614)]

### <a name="dynamic-adaptation"></a> Dynamic Adaptations

The following papers are additional readings.

- (Dynamic model selection) Lou, Wei, Lei Xun, Amin Sabet, Jia Bi, Jonathon Hare, and Geoff V. Merrett. "Dynamic-OFA: Runtime DNN Architecture Switching for Performance Scaling on Heterogeneous Embedded Platforms." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3110-3118. 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/ECV/papers/Lou_Dynamic-OFA_Runtime_DNN_Architecture_Switching_for_Performance_Scaling_on_Heterogeneous_CVPRW_2021_paper.pdf)]
- (DVFS) Tu, Fengbin, Weiwei Wu, Yang Wang, Hongjiang Chen, Feng Xiong, Man Shi, Ning Li et al. "Evolver: A deep learning processor with on-device quantization–voltage–frequency tuning." IEEE Journal of Solid-State Circuits 56, no. 2 (2020): 658-673. [[Paper](https://ieeexplore.ieee.org/document/9209075)]


## <a name="federated-ondevice-learning"></a> On-device and Federated Learning on ML Accelerators

### <a name="ondevice-learning"></a> On-device Learning

The following papers are additional readings.

- Saha, Swapnil Sayan, Sandeep Singh Sandha, and Mani Srivastava. "Machine Learning for Microcontroller-Class Hardware--A Review. In IEEE Sensors. [[Paper](https://ieeexplore.ieee.org/document/9912325)]
- Shin, Jaekang, Seungkyu Choi, Yeongjae Choi, and Lee-Sup Kim. "A pragmatic approach to on-device incremental learning system with selective weight updates." In 2020 57th ACM/IEEE Design Automation Conference (DAC), pp. 1-6. IEEE, 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9218507)]
- Chen, Chixiao, Hongwei Ding, Huwan Peng, Haozhe Zhu, Rui Ma, Peiyong Zhang, Xiaolang Yan et al. "OCEAN: An on-chip incremental-learning enhanced processor with gated recurrent neural network accelerators." In ESSCIRC 2017-43rd IEEE European Solid State Circuits Conference, pp. 259-262. IEEE, 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/8094575)]
- Chen, Xi, Chang Gao, Tobi Delbruck, and Shih-Chii Liu. "EILE: Efficient Incremental Learning on the Edge." In 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS), pp. 1-4. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9458554)]
- Lee, Jinsu, and Hoi-Jun Yoo. "An Overview of Energy-Efficient Hardware Accelerators for On-Device Deep-Neural-Network Training." IEEE Open Journal of the Solid-State Circuits Society (2021). [[Paper](https://ieeexplore.ieee.org/abstract/document/9569757)]
- Cai, Han, Chuang Gan, Ligeng Zhu, and Song Han. "Tinytl: Reduce memory, not parameters for efficient on-device learning." Advances in Neural Information Processing Systems 33 (2020): 11285-11297. [[Paper](https://proceedings.neurips.cc/paper/2020/file/81f7acabd411274fcf65ce2070ed568a-Paper.pdf)]
- Fang, Biyi, Xiao Zeng, and Mi Zhang. "Nestdnn: Resource-aware multi-tenant on-device deep learning for continuous mobile vision." In Proceedings of the 24th Annual International Conference on Mobile Computing and Networking, pp. 115-127. 2018. [[Paper](https://arxiv.org/pdf/1810.10090.pdf)]

### <a name="federated-learning"></a> Federated Learning

- Kim, Young Geun, and Carole-Jean Wu. "Autofl: Enabling heterogeneity-aware energy efficient federated learning." In MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture, pp. 183-198. 2021.. [[Paper](https://dl.acm.org/doi/abs/10.1145/3466752.3480129)]

The following papers are additional readings.

- Konečný, Jakub, H. Brendan McMahan, Daniel Ramage, and Peter Richtárik. "Federated optimization: Distributed machine learning for on-device intelligence." arXiv preprint arXiv:1610.02527 (2016). [[Paper](https://arxiv.org/pdf/1610.02527)]
- Bonawitz, Keith, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon et al. "Towards federated learning at scale: System design." Proceedings of Machine Learning and Systems 1 (2019): 374-388. [[Paper](https://proceedings.mlsys.org/paper/2019/file/bd686fd640be98efaae0091fa301e613-Paper.pdf)]
- Li, Tian, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. "Federated learning: Challenges, methods, and future directions." IEEE Signal Processing Magazine 37, no. 3 (2020): 50-60. [[Paper](https://www.researchgate.net/profile/Anit-Sahu/publication/335319008_Federated_Learning_Challenges_Methods_and_Future_Directions/links/5d8456f892851ceb791b1454/Federated-Learning-Challenges-Methods-and-Future-Directions.pdf)]
- Ruan, Yichen, Xiaoxi Zhang, Shu-Che Liang, and Carlee Joe-Wong. "Towards flexible device participation in federated learning." In International Conference on Artificial Intelligence and Statistics, pp. 3403-3411. PMLR, 2021. [[Paper](http://proceedings.mlr.press/v130/ruan21a/ruan21a.pdf)]
- Yang, Zhaoxiong, Shuihai Hu, and Kai Chen. "Fpga-based hardware accelerator of homomorphic encryption for efficient federated learning." arXiv preprint arXiv:2007.10560 (2020). [[Paper](https://arxiv.org/pdf/2007.10560.pdf)]

## <a name="industry-startups"></a> Industry Case Studies (Established Startups)

- [SambaNova] Emani, Murali, Venkatram Vishwanath, Corey Adams, Michael E. Papka, Rick Stevens, Laura Florescu, Sumti Jairath, William Liu, Tejas Nama, and Arvind Sujeeth. "Accelerating scientific applications with sambanova reconfigurable dataflow architecture." Computing in Science & Engineering 23, no. 02 (2021): 114-119. [[Paper](https://ieeexplore.ieee.org/abstract/document/9387491)]
- [Cerebreas] Lauterbach, Gary. "The Path to Successful Wafer-Scale Integration: The Cerebras Story." IEEE Micro 41, no. 6 (2021): 52-57. [[Paper](https://ieeexplore.ieee.org/abstract/document/9623424)]

The following papers are additional readings.

### <a name="industry-hardware-startups"></a> Recent Industrial ML Hardware Avenues

- (Cerebras) James, Michael, Marvin Tom, Patrick Groeneveld, and Vladimir Kibardin. "Ispd 2020 physical mapping of neural networks on a wafer-scale deep learning accelerator." In Proceedings of the 2020 International Symposium on Physical Design, pp. 145-149. 2020. [[Paper](https://dl.acm.org/doi/abs/10.1145/3372780.3380846)]
- (Numenta) Hunter, Kevin Lee, Lawrence Spracklen, and Subutai Ahmad. "Two Sparsities Are Better Than One: Unlocking the Performance Benefits of Sparse-Sparse Networks." arXiv preprint arXiv:2112.13896 (2021). [[Paper](https://arxiv.org/pdf/2112.13896.pdf)]
- (Habana Labs, now Intel) Medina, Eitan, and Eran Dagan. "Habana labs purpose-built ai inference and training processor architectures: Scaling ai training systems using standard ethernet with gaudi processor." IEEE Micro 40, no. 2 (2020): 17-24. [[Paper](https://ieeexplore.ieee.org/abstract/document/9018203)]
- (Nervana NNP-I, Intel) Wechsler, Ofri, Michael Behar, and Bharat Daga. "Spring hill (nnp-i 1000) intel’s data center inference chip." In 2019 IEEE Hot Chips 31 Symposium (HCS), pp. 1-12. IEEE Computer Society, 2019. [[Slides](https://ieeexplore.ieee.org/abstract/document/8875671)]
- (Graphcore) Jia, Zhe, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. "Dissecting the graphcore ipu architecture via microbenchmarking." arXiv preprint arXiv:1912.03413 (2019). [[Paper](https://www.graphcore.ai/hubfs/assets/pdf/Citadel%20Securities%20Technical%20Report%20-%20Dissecting%20the%20Graphcore%20IPU%20Architecture%20via%20Microbenchmarking%20Dec%202019.pdf)]
- (Groq) Abts, Dennis, Jonathan Ross, Jonathan Sparling, Mark Wong-VanHaren, Max Baker, Tom Hawkins, Andrew Bell et al. "Think fast: a tensor streaming processor (TSP) for accelerating deep learning workloads." In 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), pp. 145-158. IEEE, 2020. [[Paper](https://groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf)]
- (Mythic) MyThic AI Technology [[Webpage](https://www.mythic-ai.com/technology/)] [[Blog](https://www.mythic-ai.com/the-future-of-ai-twenty-trillion-synapses/)] [[Talks](https://www.youtube.com/watch?v=pGXWNoO5Q3M)]
- (SimpleMachines) Sankaralingam, Karthikeyan, Tony Nowatzki, Vinay Gangadhar, Preyas Shah, Michael Davies, William Galliher, Ziliang Guo et al. "The Mozart reuse exposed dataflow processor for AI and beyond: industrial product." In Proceedings of the 49th Annual International Symposium on Computer Architecture, pp. 978-992. 2022. [[Paper](https://research.cs.wisc.edu/vertical/papers/2022/isca22-mozart.pdf)]

### <a name="industry-software-startups"></a> Recent Industrial ML Software Avenues

- (NeuralMagic) Shavit, Nir. Software Architecture for Sparse ML [[Blog](https://neuralmagic.com/technology/)] [[Webinar](https://neuralmagic.com/resources/on-demand-webinars/big-brain-burnout/)]
- (MosaicML) Composer: A PyTorch Library for Efficient Neural Network Training [[GitHub](https://github.com/mosaicml/composer)] [[Docs](https://docs.mosaicml.com/en/v0.5.0/)]


## <a name="industry"></a> Industry Case Studies

- (Facebook) Anderson, Michael, Benny Chen, Stephen Chen, Summer Deng, Jordan Fix, Michael Gschwind, Aravind Kalaiah et al. "First-generation inference accelerator deployment at facebook." arXiv preprint arXiv:2107.04140 (2021). [[Paper](https://arxiv.org/pdf/2107.04140)]
- (Google) Jouppi, Norman P., Doe Hyun Yoon, Matthew Ashcraft, Mark Gottscho, Thomas B. Jablin, George Kurian, James Laudon et al. "Ten lessons from three generations shaped google’s tpuv4i: Industrial product." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 1-14. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9499913)]

The following papers are additional readings.

- (NVIDIA) Krashinsky, Ronny, Olivier Giroux, Stephen Jones, Nick Stam, and Sridhar Ramaswamy. NVIDIA A100 Ampere Architecture In-Depth, 2020. [[Blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)]
- (NVIDIA)  Andersch, Michael, Greg Palmer, Ronny Krashinsky, Nick Stam, Vishal Mehta, Gonzalo Brito and Sridhar Ramaswamy. NVIDIA H100 Hopper Architecture In-Depth, 2022 [[Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)]
- (Samsung) Jang, Jun-Woo, Sehwan Lee, Dongyoung Kim, Hyunsun Park, Ali Shafiee Ardestani, Yeongjae Choi, Channoh Kim et al. "Sparsity-aware and re-configurable NPU architecture for samsung flagship mobile SoC." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 15-28. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9499876)]
- (Tesla FSD Supercomputer) Talpes, Emil, Debjit Das Sarma, Ganesh Venkataramanan, Peter Bannon, Bill McGee, Benjamin Floering, Ankit Jalote et al. "Compute solution for tesla's full self-driving computer." IEEE Micro 40, no. 2 (2020): 25-35. [[Paper](https://ieeexplore.ieee.org/abstract/document/9007413/)]
- (IBM AI Accelerator) Fleischer, Bruce, Sunil Shukla, Matthew Ziegler, Joel Silberman, Jinwook Oh, Vijavalakshmi Srinivasan, Jungwook Choi et al. "A scalable multi-TeraOPS deep learning processor core for AI trainina and inference." In 2018 IEEE Symposium on VLSI Circuits, pp. 35-36. IEEE, 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8502276)]
- (Samsung) Lee, Sukhan, Shin-haeng Kang, Jaehoon Lee, Hyeonsu Kim, Eojin Lee, Seungwoo Seo, Hosang Yoon et al. "Hardware Architecture and Software Stack for PIM Based on Commercial DRAM Technology: Industrial Product." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 43-56. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9499894/)]
- (Samsung) Kim, Jin Hyun, Shin-haeng Kang, Sukhan Lee, Hyeonsu Kim, Woongjae Song, Yuhwan Ro, Seungwon Lee et al. "Aquabolt-XL: Samsung HBM2-PIM with in-memory processing for ML accelerators and beyond." In 2021 IEEE Hot Chips 33 Symposium (HCS), pp. 1-26. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9567191/)]
- (QualComm) Chatha, Karam. "Qualcomm® Cloud Al 100: 12TOPS/W Scalable, High Performance and Low Latency Deep Learning Inference Accelerator." In 2021 IEEE Hot Chips 33 Symposium (HCS), pp. 1-19. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9567417/)]
- (AMD) Naffziger, Samuel, Noah Beck, Thomas Burd, Kevin Lepak, Gabriel H. Loh, Mahesh Subramony, and Sean White. "Pioneering Chiplet Technology and Design for the AMD EPYC™ and Ryzen™ Processor Families: Industrial Product." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 57-70. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9499852/)]
- (IBM) Thompto, Brian W., Dung Q. Nguyen, José E. Moreira, Ramon Bertran, Hans Jacobson, Richard J. Eickemeyer, Rahul M. Rao et al. "Energy efficiency boost in the AI-Infused POWER10 processor." In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA), pp. 29-42. IEEE, 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9499822/)]
- (Xilinx) D'Alberto, Paolo, Victor Wu, Aaron Ng, Rahul Nimaiyar, Elliott Delaye, and Ashish Sirasao. "xDNN: Inference for Deep Convolutional Neural Networks." ACM Transactions on Reconfigurable Technology and Systems (TRETS) 15, no. 2 (2022): 1-29. [[Paper](https://dl.acm.org/doi/full/10.1145/3473334)]
- (ARM MLP) Bratt, Ian and John Brothers. “Arm’s first-generation machine learning processor.” In 2018 IEEE Hot Chips 30 Symposium (HCS), pp. 1-27. IEEE, 2018. [[Paper](https://old.hotchips.org/hc30/2conf/2.07_ARM_ML_Processor_HC30_ARM_2018_08_17.pdf)]


## <a name="reliability-security"></a> Reliability and Security of ML Accelerators

### <a name="reliability"></a> Reliability of ML Accelerators

- Li, Guanpeng, Siva Kumar Sastry Hari, Michael Sullivan, Timothy Tsai, Karthik Pattabiraman, Joel Emer, and Stephen W. Keckler. "Understanding error propagation in deep learning neural network (DNN) accelerators and applications." In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-12. 2017. [[Paper](https://people.csail.mit.edu/emer/papers/2017.11.sc.error_propagation_in_DNNs.pdf)]

The following papers are additional readings.

- Reagen, Brandon, Udit Gupta, Lillian Pentecost, Paul Whatmough, Sae Kyu Lee, Niamh Mulholland, David Brooks, and Gu-Yeon Wei. "Ares: A framework for quantifying the resilience of deep neural networks." In 2018 55th ACM/ESDA/IEEE Design Automation Conference (DAC), pp. 1-6. IEEE, 2018. [[Paper](https://dl.acm.org/doi/10.1145/3195970.3195997)]
- Mittal, Sparsh. "A survey on modeling and improving reliability of DNN algorithms and accelerators." Journal of Systems Architecture 104 (2020): 101689. [[Paper](https://www.sciencedirect.com/science/article/pii/S1383762119304965?casa_token=zs060XSD2NcAAAAA:HNav3CwZQCY7ip0YFRvd73KVvgPfS9huyUY58HnU2Degmc-ro0rrXLrhYky2dfuNs4Djqxi-)]
- Henkel, Jörg, Lars Bauer, Nikil Dutt, Puneet Gupta, Sani Nassif, Muhammad Shafique, Mehdi Tahoori, and Norbert Wehn. "Reliable on-chip systems in the nano-era: Lessons learnt and future trends." In 2013 50th ACM/EDAC/IEEE Design Automation Conference (DAC), pp. 1-10. IEEE, 2013. [[Paper](https://www.researchgate.net/profile/Lars-Bauer-2/publication/261164817_Reliable_On-chip_systems_in_the_nano-era_Lessons_learnt_and_future_trends/links/53cedb3c0cf2f7e53cf7e27f/Reliable-On-chip-systems-in-the-nano-era-Lessons-learnt-and-future-trends.pdf)]


### <a name="security"></a> Security of ML Accelerators

- Rouhani, Bita Darvish, M. Sadegh Riazi, and Farinaz Koushanfar. "Deepsecure: Scalable provably-secure deep learning." In Proceedings of the 55th annual design automation conference, pp. 1-6. 2018. [[Paper](https://dl.acm.org/doi/abs/10.1145/3195970.3196023)]

The following papers are additional readings.

- Darvish Rouhani, Bita, Huili Chen, and Farinaz Koushanfar. "Deepsigns: An end-to-end watermarking framework for ownership protection of deep neural networks." In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 485-497. 2019. [[Paper](https://dl.acm.org/doi/abs/10.1145/3297858.3304051)]
- Isakov, Mihailo, Vijay Gadepally, Karen M. Gettings, and Michel A. Kinsy. "Survey of attacks and defenses on edge-deployed neural networks." In 2019 IEEE High Performance Extreme Computing Conference (HPEC), pp. 1-8. IEEE, 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8916519)]
- Liu, Yuntao, Yang Xie, and Ankur Srivastava. "Neural trojans." In 2017 IEEE International Conference on Computer Design (ICCD), pp. 45-48. IEEE, 2017. [[Paper](https://arxiv.org/pdf/1710.00942.pdf)]
- Rakin, Adnan Siraj, Zhezhi He, and Deliang Fan. "Bit-flip attack: Crushing neural network with progressive bit search." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1211-1220. 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.pdf)]
- Chen, Xinyun, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. "Targeted backdoor attacks on deep learning systems using data poisoning." arXiv preprint arXiv:1712.05526 (2017). [[Paper](https://arxiv.org/pdf/1712.05526.pdf)]
- Wang, Bolun, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. "Neural cleanse: Identifying and mitigating backdoor attacks in neural networks." In 2019 IEEE Symposium on Security and Privacy (SP), pp. 707-723. IEEE, 2019. [[Paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)]
- Liu, Kang, Brendan Dolan-Gavitt, and Siddharth Garg. "Fine-pruning: Defending against backdooring attacks on deep neural networks." In International Symposium on Research in Attacks, Intrusions, and Defenses, pp. 273-294. Springer, Cham, 2018. [[Paper](https://arxiv.org/pdf/1805.12185)]
