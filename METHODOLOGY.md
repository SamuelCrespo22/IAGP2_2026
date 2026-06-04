# Metodologia: Captioning + LLM-Based Refinement

## Visão Geral

Este repositório segue uma metodologia em duas etapas para reconstruir imagens alvo a partir de prompts:

1. **Captioning inicial**: usar um modelo de captioning para gerar um prompt base a partir da imagem alvo.
2. **Refinamento por LLM**: usar um modelo de linguagem para gerar variações de prompt e avaliar qual geração melhor se aproxima da imagem alvo.

O captioning inicial funciona como uma semente. Não é necessário que seja perfeito; o objetivo é capturar a ideia central da imagem e permitir que a fase de refinamento melhore a descrição.

## Etapa 1 — Captioning inicial

- O captioning foi realizado em Colab com `Salesforce/blip-image-captioning-large`.
- Cada imagem alvo recebe um prompt base.
- Esses prompts iniciais são a base do refinamento.
- Mesmo que o prompt inicial não esteja perfeito, ele deve ser um ponto de partida razoável.

### Repetir captioning?

- Não é obrigatório repetir captioning imediatamente.
- A melhor prática é:
  1. executar uma passagem de captioning para obter o prompt base;
  2. aplicar o refinamento LLM;
  3. se a métrica não melhorar de forma consistente após `NUM_ITERATIONS`, considerar uma nova passagem de captioning com outra configuração ou modelo.
- Ou seja, o captioning deve ser utilizado como fallback quando o refinamento atual estagnar.

## Etapa 2 — Refinamento por LLM

- Para cada imagem, o loop de refinamento começa com o prompt atual mais forte.
- O LLM recebe o prompt e as métricas atuais e deve produzir exatamente 5 variações.
- Cada variação é renderizada com o mesmo setup LCM e avaliada.
- Se uma variação melhora os critérios, ela substitui o melhor prompt atual.
- Esse ciclo se repete por `NUM_ITERATIONS` dentro de cada run, e o pipeline executa múltiplas runs (`NUM_RUNS`) para explorar diferenças de seed e estabilidade.

## Métricas e seleção final

### Métrica principal: CLIP

- A seleção principal deve usar CLIP Similarity, pois ela indica se a imagem gerada corresponde ao conteúdo visual do alvo.
- Mas CLIP sozinho não garante fidelidade de textura ou pixel-level.

### Métricas de complementaridade

- `LPIPS` captura similaridade perceptual entre imagens.
- `RMSE` captura diferença pixel-a-pixel.

### Combinação de métricas

Recomenda-se uma função composta para rankear e selecionar candidatos:

1. normalizar cada métrica dentro do conjunto de candidatos:
   - `CLIP_norm = (CLIP - min(CLIP)) / (max(CLIP) - min(CLIP))`
   - `LPIPS_norm = (LPIPS - min(LPIPS)) / (max(LPIPS) - min(LPIPS))`
   - `RMSE_norm = (RMSE - min(RMSE)) / (max(RMSE) - min(RMSE))`

2. calcular um score combinado:

```text
combined_score = 0.50 * CLIP_norm + 0.25 * (1 - LPIPS_norm) + 0.25 * (1 - RMSE_norm)
```

- Alternativa válida: usar CLIP como critério principal e LPIPS/RMSE para desempate.
- O importante é não tomar a decisão final apenas com base em `CLIP_Sim`.

### Seleção Top-3

- Entre todas as runs, deve-se agrupar os candidatos e remover prompts duplicados exatos.
- Selecionar os 3 melhores com base no `combined_score`.
- Regenerar as imagens Top-3 com a mesma configuração LCM para garantir consistência.

## Pipeline determinístico do LCM

- A pipeline LCM é determinística dentro das condições definidas neste repositório:
  - a seed é extraída do nome do arquivo alvo;
  - os hiperparâmetros de renderização são fixos (`num_inference_steps=8`, `guidance_scale=8.0`, `lcm_origin_steps=50`, `768x768`).
- Isso significa que o mesmo prompt e o mesmo target filename devem gerar a mesma imagem, desde que:
  - o ambiente e as versões das bibliotecas permaneçam iguais;
  - não haja variações por algoritmos não determinísticos de CUDA.

### Nota sobre reprodutibilidade

- Para reforçar reprodutibilidade, recomenda-se:
  - fixar seeds de CPU e GPU;
  - usar `torch.backends.cudnn.deterministic = True` e `torch.backends.cudnn.benchmark = False` quando possível;
  - manter versões fixas de `torch`, `diffusers`, `transformers` e `lpips`.

## Critério de parada e thresholds

- Um critério pragmático de parada pode ser:
  - nenhuma melhoria relevante de CLIP após 2-3 iterações;
  - ou `delta(CLIP) < 0.01` em comparações consecutivas.
- Se o refinamento travar, a estratégia é tentar:
  - gerar outro prompt base com captioning alternativo;
  - ampliar a diversidade de variações do LLM;
  - ajustar pesos das métricas no score composto.

## Recomendações finais

- Manter o captioning inicial como semente, mas não como resultado final.
- Priorizar o refinamento iterativo do prompt.
- Usar métricas combinadas para reduzir falsos positivos de CLIP.
- Documentar claramente o processo de execução e os requisitos de ambiente.
- Validar a pipeline com um script que teste geração, avaliação e determinismo.
