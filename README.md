# IAGTP2: Image-to-Prompt Inversion

Este repositório contém a implementação do Projeto 2 da cadeira de Generative AI (2025/2026).

## Visão Geral

A ideia central é reconstruir imagens alvo usando um trabalho combinado de:

1. captioning inicial para obter um prompt base;
2. refinamento iterativo por LLM para melhorar o prompt;
3. avaliação de cada imagem gerada com métricas de similaridade.

O pipeline principal está em `src/main.py`, a geração LCM em `src/generator.py`, e a avaliação em `src/evaluator.py`.

## Configuração do Ambiente

1. Criar um ambiente virtual e instalar as dependências:

```bash
pip install -r requirements.txt
```

2. Copiar `.env.example` para `.env` e preencher a chave:

```text
GROQ_API_KEY=your_groq_api_key_here
```

3. Garantir que as imagens alvo estejam em `data/` ou em `tp2-chosen/` conforme o notebook.

## Execução do pipeline

1. Colocar os alvos em `data/` (por exemplo, `1159_7.png`, `7836.png`).
2. Confirmar que `GROQ_API_KEY` está configurada.
3. Executar o pipeline principal:

```bash
python -m src.main
```

4. O resultado é salvo em `outputs_15_runs/`, com:
   - `run_<i>/metrics_log.csv` para cada run;
   - `top3_final/top3_metrics.csv` para as 3 melhores prompts;
   - `global_aggregate_stats.csv` com estatísticas agregadas.

## Teste de validação

Há um script de validação da pipeline em `src/validate_pipeline.py`.

Para testar:

```bash
python src/validate_pipeline.py
```

O script valida:
- geração de imagem com LCM;
- cálculo de métricas CLIP, LPIPS e RMSE;
- determinismo reproduzível da geração;
- teste opcional de variações de prompt se `GROQ_API_KEY` estiver configurada.

## Metodologia formal

Ver `METHODODOLOGY.md` para o fluxo completo de captioning + LLM refinement.

## Notas importantes

- A pipeline LCM é configurada para ser determinística:
  - seed extraída do nome do arquivo;
  - parâmetros fixos de inference;
  - modelo `SimianLuo/LCM_Dreamshaper_v7`.
- O captioning inicial é usado como semente, não como solução final.
- A seleção final ideal deve combinar CLIP, LPIPS e RMSE em vez de usar apenas CLIP.

## Arquivos principais

- `TP2_StarterPack_Students.ipynb`: notebook inicial para Colab/VS Code Colab.
- `src/main.py`: pipeline de geração, refinamento e extração de Top-3.
- `src/generator.py`: gerador LCM.
- `src/evaluator.py`: avaliador de métricas.
- `src/validate_pipeline.py`: script de teste da pipeline.
- `METHODODOLOGY.md`: documento formal da metodologia.
- `.env.example`: exemplo de configuração de chave.
