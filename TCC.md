# TCC

[TCC - Overleaf](https://www.overleaf.com/project/6745361c19ba8b540be1761f)

## Datasets

- LVSM:
	- [Objaverse-XL](https://github.com/allenai/objaverse-xl):
		- Dataset sintético de 10M modelos 3D renderizados de posições diferentes;
		- Usado originalmente para treinar o LVSM;
			- Na verdade, o Objaverse foi usado, que é uma versão anterior com menos modelos (apenas 800K modelos);
		- Desvantagens: Tem que renderizar do zero;
			- Modifiquei os scripts para renderizar meu próprio dataset, [implementação aqui](https://github.com/gammag4/objaverse-xl).
	- [Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX):
		- Dataset com 12 poses já renderizadas dos 800K objetos do Objaverse-1.0;
		- Vantagens: Já está com as poses renderizadas, não tendo que renderizar do zero;
		- Desvantagens: Apenas 12 poses da câmera pra cada objeto e as poses não são aleatórias, sendo todas na mesma esfera de raio 2 e em posições predefinidas;
	- [RealEstate10K](https://google.github.io/realestate10k/)
		- Dataset de trechos de vídeos de Real Estate do YouTube, com câmeras mostrando várias visões das casas, com anotações das poses da câmera ao longo do vídeo;
		- Usado originalmente para treinar o LVSM;
		- Desvantagens: Tem que baixar os trechos do YouTube do zero;
			- #TODO Buscar se não já tem esse dataset (ou parte dele) baixado em algum canto;
- LVSM com componente temporal:
	- [Warwick-NTU](https://rose1.ntu.edu.sg/dataset/Warwick-NTU/):
		- Dataset com vídeos de várias câmeras de uma universidade, gravando pessoas passando pelos corredores;
		- Vantagens: Tem várias visões de câmeras e tem componente temporal;
		- Desvantagens: Domínio muito fechado (apenas corredores de uma universidade específica), câmeras são estáticas;

## Relatório

28/10/2024

- Li por cima o artigo do modelo [LVSM](https://haian-jin.github.io/projects/LVSM/);

31/10/2024

- Conversa sobre ideia inicial do TCC;
- Ideia inicial: Adicionar componente temporal ao modelo NeRF;
	- Rodar modelo NeRF localmente;
	- Buscar dataset temporal;
	- Adicionar componente temporal ao modelo existente;
	- Treinar com dataset encontrado;

01/11/2024

- Implementação:
	- Achei um código do NeRF em PyTorch que consegui alterar e rodar;
	- Rodei o modelo e renderizei um vídeo de 60 frames em 400x400. Renderizou bem em 8 horas, apesar que demoraria mais pra convergir totalmente, mas creio que 4h (talvez menos) seria o suficiente para verificar convergência do modelo, mesmo com ruído;
	- Também renderizei novamente com 6 frames em 30min pra testar, em 30min já dá para ver o modelo convergindo, apesar que com bastante ruído.

04/11/2024

- Esqueci de procurar se já havia alguma contribuição adicionando componente temporal em NeRFs;
- Achei duas, [uma adicionando componente temporal em NeRFs](https://video-nerf.github.io/) e [outra adicionando componente temporal em Gaussian Splatting](https://oppo-us-research.github.io/SpacetimeGaussians-website/);
- Li mais aprofundado o artigo do LVSM
	- Decidi que vai ser meio pesado treinar o modelo, então vou ter que usar versões mais simplificadas tanto do modelo quanto do dataset;
		- due to limitations of time and resources, we will use only synthetic datasets and only the object dataset (not the scene one)
			- and will use simplified versions of these
		- considerando q vou ter dados limitados, por enquanto e melhor testar so com 10k modelos e 128 64x64 imagens pra cada modelo
			- dai depois que fizer o modelo novo adicionando componente temporal faz a mesma coisa
			- e por ultimo se realmente tiver dado certo, testa o modelo maior com sla 100k modelos e 64 512x512 imagens, fazendo a parada de baixar 5k modelos, criar imagens dos modelos, treinar modelo com imagens e deletar
			- divide treino em batches
			- tambem usa jpg msm no lugar de png
- Mudança de planos: Adiconar componente temporal ao LVSM;
	- Buscar dataset temporal;
	- Implementar modelo original (como não tem código ainda);
	- Treinar e validar modelo;
	- Adicionar componente temporal;
	- Treinar e validar modelo temporal;

14/11/2024

- Implementação:
	- Baixei o código do Objaverse-XL, mas o código está com uns bugs (trava se colocar um dataset muito grande pra renderizar e não centraliza direito os objetos nas cenas);
	- As imagens também são bem pesadas do Objaverse, então decidi também que vou converter pra JPG por limitações técnicas, mesmo que cause artefatos no modelo;
	- Comecei a consertar o código;

17/11/2024

- Estudei plucker ray embedding;
- Estudei perceptual loss;

22/11/2024

- Tenho que ver como treinar vision transformers mais rápido;
	- Achei [esse artigo](https://arxiv.org/abs/2201.10728) falando sobre treinar vision transformers com 2040 imagens, mas ainda não li;

25/11/2024

- Implementação:
	- Terminei de ajeitar o código do Objaverse-XL que renderiza as poses a partir do dataset dos modelos 3D;
	- Está renderizando em média 1000-2000 modelos por dia, o que é bem lento;
	- Estou considerando usar um dataset com visões já renderizadas, no lugar de renderizar um meu;
		- Achei o dataset [Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX), com 12 poses renderizadas de 800000 objetos, estou considerando usar este;
			- O problema é que as poses das câmeras não são aleatórias;
			- De qualquer forma, vou usar esse dataset, vou fazer um script pra baixar e converter as imagens pra jpg;
		- tem tambem [esse dataset](https://aigc3d.github.io/gobjaverse/) q promete renderizar mais rapido
- Datasets para treino:
	- Não consegui encontrar um dataset bom que tenha dados de vídeos de cenas gravados com múltiplas câmeras se movendo com as poses destas, nem mesmo com uma câmera;
	- Achei alguns, por exemplo o do [Warwick-NTU](https://rose1.ntu.edu.sg/dataset/Warwick-NTU/), com vídeos de câmeras de uma universidade;
- Mudança de planos: No lugar de adicionar componente temporal ao modelo, irei analisar o espaço latente da versão encoder-decoder do modelo, buscando interpretar esse espaço e pensar em maneiras possivelmente de usá-lo para gerar imagens;
	- Pela dificuldade de achar/processar datasets e também pelo tempo e esforço absurdo a mais que precisaria ter;
	- Plano:
		- Fazer script para baixar dataset;
		- Implementar modelo original (como não tem código ainda);
			- Implementar versão menor também pra testar;
		- Fazer modificações e avaliar;
 
## Plano

- Estudar:
	- Ler [Training Vision Transformers with Only 2040 Images](https://arxiv.org/abs/2201.10728);
		- Procurar outros artigos nesse lado;
		- https://www.reddit.com/r/MachineLearning/comments/z088fo/r_tips_on_training_transformers/
- Dataset:
	- Olha [Objaverse-MIX](https://huggingface.co/datasets/BAAI/Objaverse-MIX) e [G-Buffer Objaverse](https://aigc3d.github.io/gobjaverse/) e vê qual compensa mais usar;
		- Provavelmente Objaverse-MIX já que já tá pronto e não tenho tanto tempo nem recursos;
	- Implementar script pra baixar dados do dataset e converter pra jpg na memória direto, para não sobrecarregar SSD;
		- Fazer o script baixar dois tipos de imanges, o original 512x512 e outra menor 64x64;
		- Entao teriam 2 datasets, um pra um transformer menor (para teste) e outro para o modelo real;
	- Vê como extrair intrínsecos e extrínsecos da câmera;
- Implementação do modelo:
	- Dar uma olhada melhor na maneira como a rede foi implementada;
		- Olha [QK-Norm](https://arxiv.org/pdf/2010.04245);
		- Olha [xFormers](https://github.com/facebookresearch/xformers) usada pra implementar modelos modernos
		- Olha [FlashAttention-2](https://arxiv.org/pdf/2307.08691) acelera treino de transformers
		- Olha [Gradient Checkpointing](https://arxiv.org/pdf/1604.06174)
		- Olha [mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
	- Implementa parte do embedding
	- Implementa parte do vision transformer (encoder-decoder);
	- Implementa modelo do perceptual loss (provavelmente VGG-19, o mesmo do LVSM);
- Treino:
	- Desativa hibernação antes de treinar modelo (não esquece);
	- Treina modelo primeiro com datasets menores com imagens menores e vai aumentando;
		- Testa primeiro com 10k objetos, depois com 100k e por ultimo com 1M;
		- Testa primeiro com 64x64 depois com 512x512;
- Avaliação:
	- Descobre como avalia os resultados do modelo (tanto pros modelos estaticos como LVSM, GS-LRM e LRM quanto pra modelos dinamicos como os modelos temporais NeRF e 3DGS la);
	- PSNR, perceptual loss, etc;
- Análise do espaço latente:
	- Analisar espaço latente da versão encoder-decoder do LVSM: Fazer modificações no espaço latente e analisar diferenças entre resultado original e modificado no decoder;
	- Modificações pra fazer:
		- Modificar valores em tokens específicos;
		- Adicionar ou remover tokens na sequência;
		- Alterar ordem dos tokens;
		- Interpolar entre dois espaços latentes para tentar chegar de uma cena a outra;
			- Pra interpolar entre espaços latentes com número diferente de tokens, teria que achar o token identidade, que seria um token que quando adicionado a sequência não altera a saída;
			- Daí adicionaria várias vezes esse token a sequência menor e daí com sequências de mesmo tamanho daria pra interpolar;

Plano antigo (LVSM temporal):

- criar processo pra ir baixando e renderizando modelos (ate pelo menos 100k modelos, 32 imgs pra cada dando 3.2M imagens)
	- testa com 10k primeiro, sendo 320k imagens
	- cria dois sets de imagens pros modelos, um 512x512 e outro 64x64
- implementar modelo
	- cria modelo que gera embeddings a partir das imagens e dados de camera
	- cria vision transformer
	- cria modelo de perceptual loss
	- cria todas funcoes de perda no processo
- treina modelo com imagens
	- treina primeiro com imagens pequenas
	- depois com imagens maiores
- adiciona componente temporal no modelo e as funcoes de loss temporais (seguindo os modelos temporais do NeRF e 3DGS la)
- cria dataset novo
	- descobre como datasets foram gerados
	- cria dataset temporal (do mesmo jeito dos que foram gerados pra poder usar scripts existentes do blender)
- treina modelo novo
- descobre como avalia os resultados do modelo (tanto pros modelos estaticos como LVSM, GS-LRM e LRM quanto pra modelos dinamicos como os modelos temporais NeRF e 3DGS la)
 - avalia modelo novo
