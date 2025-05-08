# TCC-Fake_DICOM

Orientador:
> Prof. Dr. Paulo Sergio Silva Rodrigues

Alunos:
> Carlos Massato Horibe Chinen (R.A.: 22.221.010-6)
> 
> Gabriel Nunes Missima (R.A.: 22.221.040-3)
> 
> Vinicius Alves Pedro (R.A.: 22.221.036-1)

> [!WARNING]
> **BASE**:
> 
> * *Bases*<sub>(pasta)</sub> > Bases de dados utilizadas para testar os códigos
> * *DATASET_dicom_fourier_shift*<sub>(pasta)</sub> + *DATASET_dicom_fourier_spectrum*<sub>(pasta)</sub> > Bases geradas em *Criptografia*<sub>(pasta)</sub> para rodar *testes_CNN_base_Criptografia*<sub>(pasta)</sub>

> [!IMPORTANT]
> **Criptografia**<sub>(pasta)</sub>:
> 
> * *rsa_sha3/rsa_pixel_crypto_metrics.ipynb* > Código para testar a criptografia utilizando o algoritmo RSA no Pixel de cada Slice de uma imagem DICOM e calculando o tempo médio de execução
> * *rsa_sha3/rsa_pixel_decrypt_metrics.ipynb* > Código para testar a descriptografia utilizando o algoritmo RSA no arquivo criptografado do Pixel de cada Slice de uma imagem DICOM e calculando o tempo médio de execução
> * *rsa_sha3/rsa_spectrum_crypto_metrics.ipynb* > Código para testar a criptografia utilizando o algoritmo RSA no espectro de Fourier de cada Slice de uma imagem DICOM e calculando o tempo médio de execução
> * *rsa_sha3/rsa_spectrum_decrypt_metrics.ipynb* > Código para testar a descriptografia utilizando o algoritmo RSA no arquivo criptografado do espectro de Fourier de cada Slice de uma imagem DICOM e calculando o tempo médio de execução

> [!IMPORTANT]
> **Rede Neural Convolucional** (CNN):
> 
> * *CNN-ResNet50-official.ipynb* > Código da CNN aplicada em espectros de altas frequências das imagens - última versão
> 
>   ┕╸Lê base de dados; Simula imagens com informações maliciosas; Define CNN com arquitetura ResNet50; Efetua o treinamento
> 
> * *All_Heatmap.ipynb*: Código para gerar Mapas de Ativação referentes ao desempenho do modelo CNN
> 
> * *Any_Percent.ipynb*: Código para gerar resultado médio de vários folds
> 
> * *Official Results*<sub>(pasta)</sub>: Apresenta: Modelos salvos; Resultados dos k-folds; Resultados dos Heatmaps
>
> * *testes_CNN_base_Criptografia*<sub>(pasta)</sub>: Apresenta teste de modelo CNN feita a partir da base gerada wm *Criptografia*<sub>(pasta)</sub> e seus resultados
>
> * *testes_CNN_base-Yildirim*<sub>(pasta)</sub>: Apresenta testes de modelos CNN anteriores e seus resultados

> [!TIP]
>   Novas metodologias de prevenção a técnicas de intrusão é um dos problemas mais requisitados na área médica, sobretudo em sistemas de hospitais, consultórios, clínicas, entre outros. Um dos maiores desafios dessas tarefas refere-se ao armazenamento e compartilhamento de imagens no formato Digital Imaging and Communications in Medicine (DICOM), para solucionar incidentes de ataques cibernéticos que propaguem ransomware e malware em arquivos DICOM. Embora a aplicação de algoritmos de criptografia seja muito comum, os modelos têm se mostrado insuficientes para garantir a integridade e autenticidade das imagens médicas, falhando em detectar, com precisão, alterações maliciosas em exames DICOM. Para solucionar esse problema, neste trabalho é proposto o uso de redes neurais profundas para avaliar se a combinação de algoritmos de criptografia com transformadas temporais aumenta a segurança de imagens DICOM contra tentativas intrusão maliciosas. A partir da metodologia proposta, é esperado obter um modelo de segurança em imagens médicas capaz de dificultar a introdução de conteúdo malicioso, e identificar tais imagens comprometidas em bases de dados específicas.
