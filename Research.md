## Transformer

- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
	- #TODO
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
	- Pre-LN: Pre-Layer Normalization;
		- LayerNorm is put inside the residual block, right before self-attention layer;
		- Better, because it doesn't need the warmup stage;
	- Post-LN: Post-Layer Normalization;
		- LayerNorm is put between the residual blocks (original way from Attention is All You Need);

## SIRENs

#TODO incomplete

- SRNs: https://arxiv.org/pdf/1906.01618
- SIRENs: https://arxiv.org/pdf/2006.09661
	- https://www.youtube.com/watch?v=Q2fLWGBeaiI
- LLFF: https://arxiv.org/pdf/1905.00889
- Neural Volumes: https://arxiv.org/pdf/1906.07751

## NeRF

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934)

- problem: view synthesis
- continuous 5D function (radiance emitted in each direction $(\theta, \phi)$ at each point $(x, y, z)$)
- uses deep fully connected neural network (MLP, no CNN)
- it is not just recreating the 3d scene, it also takes into account specularity and other surface properties
	- it handles transparent objects
	- so from this, you can get material maps and light sources #TODO Check this or create model for this
- NeRF is different from traditional neural networks, because instead of using training and test data, we overfit each neural network to a specific scene #TODO put in NN techniques
	- so each scene has its own neural network where the scene is embedded in the weights
- the whole process is differentiable
- uses standard sampling to sample uniformly along the rays
	- original:
		- $C(\mathbf{r}) = \int_{t_{n}}^{t_{f}} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$, where $T(t) = e^{-\int_{t_{n}}^{t} \sigma(\mathbf{r}(s)) \, ds}$;
	- after sampling:
		- $t_{i} \sim \mathcal{U}\left[ t_{n} + \frac{i - 1}{N}(t_{f} - t_{n}), t_{n} + \frac{i}{N}(t_{f} - t_{n}) \right]$;
		- $\overset{ \circ }{ C }(\mathbf{r}) = \sum_{i = 1}^{N} T_{i} (1 - e^{-\sigma_{i} \delta_{i}}) \mathbf{c}_{i}$, where $T_{i} = e^{- \sum_{j = 1}^{i - 1}\sigma_{j}\delta_{j}}$;
- uses positional encoding (that sine and cosine encoding thing, originally from transformers)
	- #TODO This positional encoding makes it better for the network to understand absolute positions. is there such a technique for relative positions? (relative to what?) #TODO get both techniques and put them in NN techniques
	- uses:
		- in transformers, they were used to encode discrete positions into a continuous space
		- here they are used to allow the network to learn better about the positions
		- in both cases, they allow the network to have a better understanding of the positions by giving different levels of subdivision of the position
	- tested with 10 and 4 layers
- Differences from old CNN models: It is learned from points and viewing angles in the scene, so it is consistent when moving around the scene
	- it is geometric consistent
- It doesn't need camera position and direction because it uses the position of the point being viewed, not of the camera

Network:

- Inputs: point in scene and viewing direction (direction where the camera is viewing from);
	- $(\mathbf{x}, \mathbf{d}) = (x, y, z, \theta, \phi)$;
- Outputs: Color and density (is there something there?);
	- $(\mathbf{c} , \sigma)$;
- Process:
	- First it uses some traditional algorithm to find the positions of the cameras in the scene for each image;
	- Loss: Then it uses this to sample points in the scene, make the network output the density and color at that specific point and compare it to the actual density and color from the original scene images;
		- It samples by sending rays from the camera settings found from the traditional algorithm and sampling points along these rays;
- the network is encouraged to be multiview consistent by making it predict $\sigma$ only from position (independent from viewing direction) and $\mathbf{c}$ from both position and viewing direction;

## NeRF-W

[NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections](https://arxiv.org/pdf/2008.02268)

- Inputs: Viewpoint, appearance embedding, transient embedding;
	- From viewpoint and appearance embedding, we reconstruct the static scene (the part of the scene that is permanent);
	- Adding transient embedding, we reconstruct the transient scene (the part that is transient);
- Loss: Then we composite both scenes to generate the final image and minimize error by comparing to the target image;
- uncertainty ???
- #TODO does it use one transient scene for each image and the same static scene for all images?
- can change appearance of scene by changing appearance embedding
	- the model generates scenes for all appearances instead of just one general scene
	- can interpolate between embeddings to find appearances in between
- it is more geometric consistent than NeRF, because it can handle different appearances and transients

## LVSM

[ LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias](https://haian-jin.github.io/projects/LVSM/)

- talks about previous works that worked on a per-scene basis (3DGS, NeRF, and their variants)
- talks about generalizable frameworks that estimate representations or generate novel images in a feed-forward manner often using additional 3d inductive biases
- their problems: 3d inductive biases limit model flexibility

LRMs: Large Reconstruction Models
- made progress in removing architecture-level bias (like epipolar geometry or plane-sweep volumes)
- problem: still rely on representation-level biases (like NeRFs, meshes or 3DGS, along with their rendering equations)

LVSM
proposal: minimize 3D inductive biases and use a fully data-driven approach to NVS
- more accurate, efficient and scalable with photorealistic quality
uses: transformer-based approach that does NVS from posed sparse-view inputs

architecture:

- $p$: patch size (length of the side of each patch);
	- $p = 16$;
- $d$: Latent size (input token latent space dimension);

network:

- tokenizing input images
	- breaks the image into patches of $p \times p \times 3$, where 3 are the RGB channels
	- computes plucker ray embeddings for each pixel
	- creates plucker ray patches of $p \times p \times 6$, where 6 are the 6 variables of each plucker ray #TODO
	- $x_{ij} = \mathrm{Linear}_{input}([\mathbf{I}_{ij}, \mathbf{P}_{ij}]) \in \mathbb{R}^{d}$
		- concatenates both patches
		- reshapes the concatenated array into a 1D vector
		- maps (projects) it into an input patch token $x_{ij} \in \mathbb{R}^{d}$
- tokenizing target image
	- same as input images, but uses only the plucker ray embeddings to get the resulting image
	- $\mathbf{q}_{j} = \mathrm{Linear}_{target}(\mathbf{P}_{j}^{t}) \in \mathbb{R}^{d}$
		- reshapes plucker ray patches into a 1D vector
		- maps (projects) it into an input patch token $x_{ij} \in \mathbb{R}^{d}$
- in both input and target image tokenizations, the plucker ray embeddings ($\mathbf{P}_{ij}$ or $\mathbf{P}_{j}^{t}$) are computed from the camera poses and intrinsics
- LVSM synthesizes the new view using a full transformer model:
	- $y_{1}, \dots, y_{l_q} = M(q_{1}, \dots, q_{l_q} | x_{1}, \dots, x_{l_x})$

- uses only original transformer encoder layers in both encoder and decoder (no masking in self-attention layers)
- increases number of decoder layers
- also removes cross-attention layers

plucker ray embedding/plucker coordinates:

- [Julius Plucker. Xvii. on a new geometry of space. Philosophical Transactions of the Royal Society of London, pp. 725–791, 1865.](https://royalsocietypublishing.org/doi/pdf/10.1098/rstl.1865.0017)
	- paper originally referenced in LVSM paper
- a way to represent lines in 3D space
- are coordinates for lines in 3D space
- these are like coordinates for a directed line in space
	- given two points $A, B$ and a line directed from $A$ to $B$, the plucker coordinates of the directed line are given by $L = (B - A; B \times A)$
		- [source](https://realtimecollisiondetection.net/blog/?p=13)
	- another way is to use a direction vector $\mathbf{l}$ and a point $\mathbf{p}$ that the line passes through, where the plucker coordinates will be $L = (\mathbf{l}, \mathbf{m})$, where $\mathbf{m} = \mathbf{p} \times \mathbf{l}$ is the moment vector of the line
		- [source](https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf)
		- in my implementation, i will use this representation, using $\mathbf{p}$ as the camera origin in space and the direction $\mathbf{l}$ as $\mathbf{l} = \mathbf{q_{ij}} - \mathbf{p}$, where $\mathbf{q_{ij}}$ is the coordinate of the pixel $ij$ of the image in 3D space
			- although the choice of $\mathbf{p}$ is arbitrary, since in the paper, it shows that $\mathbf{m}$ is independent of the chosen $\mathbf{p}$
- it has 6 coordinates for each pixel (for the line passing through it)
- #TODO still dont understand exactly why is this representation used instead of any other (in this specific paper and in computer graphics in general), maybe is because it gives a continuous representation for lines instead of a discontinuous one, allowing the model to perform better, and in general, allowing errors to give more precise results (because it is continuous)?
	- it maps each line in 3D into a point in a 6D space
		- so it looks like the model is actually just learning a function that maps each 6D coordinate to a specific RGB color
		- but it is doing more than that, because it pays attention to different coordinates (it creates associations between ray embeddings in space)
	- but why specifically that embedding and not just choosing two arbitrary points in the line?
		- maybe for two reasons
		- first, it is independent of the chosen points, being an embedding of the line that does not depend on any point from that line
			- and also because it doesn't depend on arbitrary points, the notion of closeness/distance between the points in the 6D space will be global and uniform
			- it will have a topology that can have a metric distance
		- second, it allows the embedding to be a continuous representation
			- #TODO find cases that would be discontinuous with a normal embedding of two points?
		- basically that space has a better topology
- #TODO note that plucker coordinates (this referenced here) is just a special case of plucker embedding

perceptual loss:

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155)
	- this is the paper originally referenced in LVSM paper
	- this paper also uses [VGG-19](https://arxiv.org/pdf/1409.1556), just like the paper below
- [GS-LRM](https://arxiv.org/pdf/2404.19702) perceptual loss:
	- [in that paper](https://arxiv.org/pdf/2404.19702), it says that the perceptual loss from [here](https://arxiv.org/pdf/1707.09405) based on [VGG-19](https://arxiv.org/pdf/1409.1556) is empirically found in their tests to be more stable than [LPIPS](https://arxiv.org/pdf/1801.03924)
	- GS-LRM also references [LRM](https://arxiv.org/pdf/2311.04400) (which was also referenced by LVSM) that uses LPIPS
- [LRM](https://arxiv.org/pdf/2311.04400) perceptual loss:
	- uses LPIPS

training:

- Uses [QK-Norm](https://arxiv.org/pdf/2010.04245) to prevent exploding gradients problem;

rendering images:

- object-level datasets:
	- uses [objaverse 1.0](https://objaverse.allenai.org/objaverse-1.0/), rendering 32 random views for 730k objects
	- it renders using the same settings used in [GS-LRM](https://arxiv.org/pdf/2404.19702)
		- follows same settings used in [LRM](https://arxiv.org/abs/2311.04400)
			- normalizes the shape to the box $[-1, 1]^{3}$ and renders 32 random views with a same camera pointed towards the shape at arbitrary poses
			- renders images of resolution 1024x1024
			- camera poses for each view are sampled from a ball with radius in range $[1.5, 3.0]$ and with height in range $[-0.75, 1.60]$
				- what does this mean, does it mean that the center of the ball has a height within $[-0.75, 1.60]$?
			- images are rendered with a pure white background
			- processed in total 730648 3D assets
			- to evaluate model, uses new images generated from objaverse
		- differences from LRM:
			- uses ball with radius in $[1.5, 2.8]$
			- renders images of resolution 512x512
		- only renders views, uses no other information from models such as depth
- scene-level datasets: #TODO 
	- Uses [RealState10K](https://google.github.io/realestate10k/)
	- There is supposedly a better way to download, using the script [here](https://github.com/Findeton/real-state-10k).

rendering process:

- we used objaverse-xl, modifying the scripts found in https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering#-objaverse-xl-rendering-script
- #TODO we will actually create a script to download from Objaverse-MIX

## Referências

- modelos
	- transformers
		- [Attention is All You Need (140000 citations, jun 2017)](https://arxiv.org/abs/1706.03762)
			- started transformers
	- vision transformer
		- [ViT (47182 citations, oct 2020)](https://paperswithcode.com/method/vision-transformer)
			- started vision transformers
	- nerf
		- [NeRF (9150 citations, mar 2020)](https://www.matthewtancik.com/nerf)
			- started the new NVS era
		- [NeRF in the Wild (1415 citations, aug 2020)](https://nerf-w.github.io/)
		- [NeRF-XL (code coming, 2 citations, apr 2024)](https://research.nvidia.com/labs/toronto-ai/nerfxl/)
		- [FrugalNeRF (code coming, 0 citations, oct 2024)](https://arxiv.org/pdf/2410.16271)
	- gaussian splatting
		- [3DGS (1768 citations, aug 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
			- started gaussian splatting
		- [pixelSplat (85 citations, apr 2024)](https://davidcharatan.com/pixelsplat/)
		- [MVSplat (44 citations, jul 2024)](https://donydchen.github.io/mvsplat/)
		- [Splatt3R (1 citation, aug 2024)](https://splatt3r.active.vision/)
	- nerf + transformer
		- [LRM (no code, 176 citations, nov 2023)]()
			- started all the nerf/gs + transformer models
		- [M-LRM (1 citation, jun 2024)](https://murphylmf.github.io/M-LRM/)
	- gs + transformer
		- [Triplane Meets Gaussian Splatting (88 citations, dec 2023)](https://zouzx.github.io/TriplaneGaussian/)
		- [GS-LRM (no code, 6 citations, apr 2024)](https://sai-bi.github.io/project/gs-lrm/)
		- [TranSplat (code coming, 2 citations, aug 2024)](https://xingyoujun.github.io/transplat/)
	- pure transformer
		- [LVSM (code coming, oct 2024)](https://haian-jin.github.io/projects/LVSM/)
			- started pure transformer models
	- space time (nerf/gs)
		- [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/)
		- [Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis](https://oppo-us-research.github.io/SpacetimeGaussians-website/)
	- diffusion #TODO
		- [CAT3D](https://cat3d.github.io/)
- datasets:
- other:
	- nerf + efficiency
		- improving efficiency of neural radiance fields
		- https://arxiv.org/pdf/2206.00878
		- https://arxiv.org/html/2312.11537v2
		- neural radiance fields efficiency review
		- https://arxiv.org/html/2306.03000v3
	- content generation/latent space generation
		- latent space nerf generation
		- https://arxiv.org/abs/2211.07600
