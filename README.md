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
> * Bases > Bases de dados utilizadas para testar os códigos

> [!IMPORTANT]
> * CNN-espectro.ipynb > Código base para testar CNN em espectros de altas frequências das imagens
>
> Lê base de dados; Separa imagens em grupos para que não haja enviesamento da base; Simula imagens com informações maliciosas; Apresenta uma CNN base (ainda a definir arquitetura final) e mostra seus resultados.
>   
> * CNN-espectro-Resnet152-1.ipynb > Código para testar a arquitetura Resnet152 em 5% aleatório da área do espectros de altas frequências das imagens com 5% de ruído, utilizando método Monte-Carlo
> * CNN-espectro-Resnet152-2.ipynb > Código para testar a arquitetura Resnet152 em 12% total da área do espectros de altas frequências, utilizando método Monte-Carlo
> * CNN-espectro-Resnet50-3.ipynb > Código para testar a arquitetura Resnet50 em 12% total da área do espectros de altas frequências, utilizando método Monte-Carlo
> * CNN-espectro-Resnet50-4.ipynb > Código para testar a arquitetura Resnet50 em 12% total da área do espectros de altas frequências, utilizando método 5-cross-Fold (Apresenta erro de memória no K-Fold)
> * CNN-espectro-Resnet50-5.ipynb > testes ...
> * Frequencias.ipynb > Código para testar a geração de espectros de altas e baixas frequências

> [!TIP]
> * Classification_report > Avaliações de desempenho geradas por CNN-espectro.ipynb
> * Confusion_matrix > Matrizes de confusão geradas por CNN-espectro.ipynb
> * Convergence_graphs > Gráficos de convergência gerados por CNN-espectro.ipynb
> * Groups > Grupos de imagens gerados por CNN-espectro.ipynb
> * Models_saves > Modelos gerados por CNN-espectro.ipynb
> * output_images > Imagens das frequências geradas por Frequencias.ipynb
