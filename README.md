# Video Compression using Generative Models

Currently, the amount of video traffic is growing exponentially and so is the availability of high-resolution videos from mobile devices. However, this also increases the storage burden and bandwidth requirements to transfer these videos. So video compression techniques are crucial for video transfer, processing, and storage. In recent literature, deep learning has introduced a new line of thinking in end-to-end learned image compression. These works generally involve learning a generative network jointly with a prior distribution network by maximizing the ELBO on the image likelihood. However, adapting these methods from a single image to the video domain is still an open problem. Moreover, utilizing the potential of deep learning approaches for video compression on real-world mobile devices is an immense challenge due to the computational complexity of deep networks and the limitations of mobile hardware.
We designed a system to perform video compression and reconstruction to overcome high-resolution video storage and transfer issues. The information richness of the reconstructed video should match the original video. The primary input and output are both high-resolution videos as demonstrated in Figure ~\ref{fig:genPipe}. The only difference is that the output video is a reconstructed high-resolution video generated from the compressed video. Since high-resolution videos contain lots of helpful information, the compression-reconstruction process should keep as much original information as possible, and the reconstructed videos should be able to be used for downstream tasks and meet industrial requirements.

## Folder Structure

    .
    ├── HSTVC                 # code for Hybrid Spatio Temporal Video Compression
    ├── HyperRIM              # modified code for IMLE adapted from Generating Unobserved Realities
    ├── codes                 # codes for SelfC, data pre-processing and data analysis
    │   ├── models            # encoder and decoder modules for SelfC       
    │   └── options           # train and test configs for SelfC
    ├── env.yml               # anaconda environment file for SelfC
    └── ...

HSTVC and HyperRIM have their individual Readmes for their individual projects
