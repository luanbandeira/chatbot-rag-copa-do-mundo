# Comparação zero-shot vs. few-shot no prompt de geração

Atividade Parte 2 — Engenharia de Prompt, item 5 (testar um prompt em duas
versões) e item 6 (validação da saída estruturada).

Todos os números deste documento vêm de execução real contra a API da Groq em
29/09/2026. Os dados brutos estão em `docs/dados/`.

## 1. Método

O prompt avaliado é o de **geração da resposta**
(`GENERATE_ANSWER_SYSTEM_PROMPT` + `build_generate_answer_messages`), único
da aplicação que já expõe as duas variantes através do parâmetro `few_shot`.

Para isolar a variável, o script `backend/testes/comparar_zero_few_shot.py`:

1. recupera o contexto do índice FAISS **uma vez** por pergunta;
2. monta as mensagens duas vezes a partir desse mesmo contexto — uma com
   `few_shot=False`, outra com `few_shot=True`;
3. chama a LLM nas duas versões com a mesma temperatura.

Isso é mais controlado do que subir o servidor duas vezes trocando
`GENERATE_FEW_SHOT` no `.env`: as duas variantes recebem exatamente o mesmo
contexto, então qualquer diferença observada vem do prompt.

| Parâmetro | Valor |
|---|---|
| Modelo | `openai/gpt-oss-20b` (Groq) |
| Temperatura | 0.2 (a mesma de `generate_answer_node`) |
| Histórico | vazio nos dois lados |
| Chunks recuperados | 8 (`TOP_K`) |
| Índice | 26 documentos, 1228 chunks, `intfloat/multilingual-e5-small` |

A variante few-shot injeta os dois pares de `GENERATE_ANSWER_EXAMPLES`: um
caso factual respondido a partir do contexto e um caso de contexto
insuficiente cuja resposta esperada é a frase de abstenção.

## 2. Resultados

`fr` = frases, `ch` = caracteres. O prompt limita a resposta a 3 frases.

| Caso | Zero-shot | Few-shot |
|---|---|---|
| **factual-simples**<br>"Quem venceu a Copa do Mundo de 1970?" | 1 fr, 40 ch<br>"O Brasil venceu a Copa do Mundo de 1970." | 1 fr, 100 ch<br>"O Brasil venceu a Copa do Mundo de 1970, derrotando a Itália por 4 a 1 na final disputada no México." |
| **factual-com-detalhe**<br>"Onde foi disputada a final de 2014 e quem ganhou?" | 3 fr, 157 ch<br>Maracanã; Alemanha × Argentina; Alemanha campeã. | 2 fr, 180 ch<br>Maracanã; Alemanha campeã pela quarta vez. |
| **detalhe-ausente**<br>"Quantos espectadores na final de 1950, número exato?" | 1 fr, 21 ch<br>"199.854 espectadores." | 2 fr, 135 ch<br>"...contou com 199.854 espectadores, oficialmente registrados. Desses, 173.850 foram pagantes." |
| **fora-da-base**<br>"Placar da final da Champions de 2019?" | 1 fr, 49 ch<br>**Abstenção** (frase canônica) | 1 fr, 49 ch<br>**Abstenção** (frase canônica) |
| **pede-resposta-longa**<br>"Resumo completo e detalhado de toda a história das Copas" | 2 fr, 212 ch<br>"Desculpe, mas a base consultada não contém informações suficientes..." | 1 fr, 49 ch<br>**"Não encontrei essa informação na base consultada."** |
| **conhecimento-proprio**<br>"Quantos títulos o Brasil tem e em que anos?" | 2 fr, 94 ch<br>Cinco títulos: 1958, 1962, 1970, 1994, 2002. | 1 fr, 95 ch<br>Cinco títulos: 1958, 1962, 1970, 1994, 2002. |

### Aderência ao limite de 3 frases

**Empate: 0/6 violações nas duas variantes.** A regra textual do system prompt
já é suficiente; os exemplos não acrescentaram nada aqui. Vale notar que o
caso `pede-resposta-longa`, desenhado justamente para tentar furar o limite,
não furou em nenhuma das duas — mas por um motivo diferente do esperado: em
vez de escrever um texto longo, as duas variantes se recusaram a responder.

### Uso exclusivo do contexto

**Empate.** Nenhuma das duas variantes trouxe fato que não estivesse no
contexto recuperado. Em `conhecimento-proprio`, que foi montado como armadilha
(a LLM certamente "sabe" os títulos do Brasil de cor), as duas responderam
exatamente os cinco anos presentes na base, sem acrescentar comentário próprio.

### Comportamento diante de contexto insuficiente

**Aqui o few-shot ganha.** Nos dois casos em que a resposta correta era se
abster:

- `fora-da-base`: as duas produziram a frase canônica. Empate.
- `pede-resposta-longa`: o few-shot devolveu exatamente
  `"Não encontrei essa informação na base consultada."`, enquanto o zero-shot
  parafraseou — *"Desculpe, mas a base consultada não contém informações
  suficientes para responder detalhadamente à sua pergunta. Os dados
  disponíveis incluem apenas recordes de participações e algumas curiosidades
  específicas."*

As duas recusas estão comportamentalmente corretas. A diferença é de
padronização: o segundo exemplo do prompt ensina a formulação exata, e o
few-shot a reproduz. O zero-shot inventa uma formulação nova a cada vez e,
de quebra, vaza um detalhe sobre a composição da base que o usuário não pediu.

### Verbosidade

O few-shot puxa para respostas **mais completas em uma frase só**. O contraste
mais claro é `detalhe-ausente`: o zero-shot respondeu com um fragmento de 21
caracteres ("199.854 espectadores."), sem sequer repetir do que se tratava; o
few-shot devolveu uma frase inteira, contextualizada, com o dado extra dos
pagantes — que também está no contexto.

## 3. Teste de estabilidade

Como a temperatura não é zero, a mesma pergunta pode gerar respostas
diferentes. Rodamos 2 casos, 3 vezes cada, em cada variante
(`--repeticoes 3`), contando quantas respostas **distintas** saíram:

| Caso | Zero-shot | Few-shot |
|---|---|---|
| factual-simples | 3 respostas distintas em 3 | **1 resposta distinta em 3** |
| conhecimento-proprio | 3 respostas distintas em 3 | 3 respostas distintas em 3 |

A leitura ingênua seria "few-shot estabiliza a saída". **Não é isso, e vale
registrar o porquê.**

Em `factual-simples` o few-shot devolveu, nas três execuções, a string:

> "O Brasil venceu a Copa do Mundo de 1970, derrotando a Itália por 4 a 1 na
> final disputada no México."

Que é, **caractere por caractere, a resposta do primeiro exemplo em
`GENERATE_ANSWER_EXAMPLES`** — e a pergunta de teste
("Quem venceu a Copa do Mundo de 1970?") é, também literalmente, a pergunta
desse mesmo exemplo. O modelo não está seguindo melhor a instrução: está
copiando o exemplo que tem na frente.

Em `conhecimento-proprio`, uma pergunta que não se parece com nenhum exemplo,
o few-shot variou tanto quanto o zero-shot (3 de 3).

**Conclusão:** neste conjunto, os exemplos não reduziram a variabilidade das
respostas de forma generalizável. A estabilidade observada foi um artefato de
contaminação entre o caso de teste e o exemplo do prompt.

## 4. Validação da saída estruturada (item 6)

O nó `classify_question` pede um JSON que **não é exibido ao usuário**: é
consumido por `graph.route_after_classify` para escolher a rota do grafo. Um
JSON malformado, portanto, não é problema cosmético — muda o caminho da
conversa. `parse_classify_output` tem fallback para `DOMINIO`/`BAIXA`, e o
risco é esse fallback estar mascarando falhas silenciosamente.

`backend/testes/validar_classificacao.py` roda 18 entradas e inspeciona o
texto **bruto** devolvido pela LLM, antes do parse — única forma de detectar o
fallback, já que depois dele um JSON quebrado vira indistinguível de uma
classificação `DOMINIO` legítima.

| Métrica | Resultado |
|---|---|
| Classificação correta | **17/18** |
| JSON bruto inválido | **0/18** |
| Fallback acionado | **0/18** |

As 18 entradas incluem perguntas bem formadas, capitalização irregular
("QUEM GANHOU A COPA DE 1986???"), erro de digitação ("qeum venceu"), frase
incompleta ("e a de 2010"), saudações, pedidos fora do domínio e duas
tentativas de injeção direta.

O contrato de saída estruturada se sustentou em 100% dos casos. As duas
injeções diretas foram classificadas como `FORA_DOMINIO`, ou seja, barradas
antes de qualquer busca ou geração.

**A única divergência** foi a entrada `"artilheiro"` (palavra solta),
classificada como `FORA_DOMINIO` quando o esperado era `DOMINIO`. É
defensável — a palavra sozinha é ambígua —, mas significa que um
acompanhamento de uma palavra só é rejeitado em vez de resolvido pelo
histórico.

## 5. Achados que afetam outras partes da atividade

### 5.1. O gate de evidência por distância nunca dispara

`EVIDENCE_DISTANCE_THRESHOLD` está em **1.3**, mas as distâncias L2 reais
medidas neste índice são:

| Tipo de pergunta | Distância medida |
|---|---|
| Coberta pela base | 0,163 – 0,201 |
| Sobre Copa, detalhe não coberto | 0,200 – 0,312 |
| Totalmente fora do domínio | 0,354 – 0,388 |

Nada chega perto de 1,3, então `has_evidence` é sempre verdadeiro e a rota
`respond_no_evidence` é inalcançável na prática.

A causa está no próprio comentário do `config.py`, que diz ter calibrado
observando 0,6–1,1 para chunks relevantes e >1,3 para irrelevantes — valores
do `paraphrase-multilingual-MiniLM-L12-v2`. Mas `app/embeddings.py` usa
`intfloat/multilingual-e5-small`, que normaliza os vetores e comprime as
distâncias numa faixa bem mais estreita.

**Baixar o limiar não resolve sozinho.** "Quantos espectadores exatos na final
de 1950" dá 0,1998 e "Qual seleção ganhou a Copa de 2022" dá 0,2005 — quase
idênticos. O retriever encontra os chunks certos sobre 1950; eles apenas podem
não conter o número pedido. Distância mede **assunto**, não **cobertura do
fato**. Separar as duas coisas é trabalho do prompt de geração (que manda
abster) e do nó de verificação, não do limiar.

Um limiar em torno de 0,32–0,34 separaria fora-de-domínio de dentro-do-domínio,
mas essa função já é exercida antes, pelo classificador.

### 5.2. Exemplo do prompt contaminado pelo caso de teste

Como detalhado na seção 3, `GENERATE_ANSWER_EXAMPLES[0]` usa exatamente a
pergunta "Quem venceu a Copa do Mundo de 1970?". Qualquer avaliação que use
essa pergunta mede memorização, não qualidade do prompt. Recomenda-se trocar o
exemplo para uma edição que não seja usada nos testes.

## 6. Conclusão

| Critério | Vencedor |
|---|---|
| Limite de 3 frases | Empate (0/6 violações dos dois lados) |
| Uso exclusivo do contexto | Empate (nenhum vazamento dos dois lados) |
| Padronização da abstenção | **Few-shot** |
| Completude da resposta curta | **Few-shot** |
| Estabilidade entre execuções | Empate (o ganho aparente era contaminação) |
| Custo em tokens | **Zero-shot** (não carrega os exemplos) |

O few-shot **não corrigiu nenhum erro de comportamento** — o zero-shot já
respeitava o limite de frases, já usava só o contexto e já se abstinha quando
devia. O que os exemplos fizeram foi **padronizar a forma**: a frase exata da
abstenção e um pouco mais de completude nas respostas curtas.

Para esta aplicação, isso justifica manter `GENERATE_FEW_SHOT=true` quando a
consistência do texto exibido importar, e `false` para economizar tokens, já
que a correção das respostas não muda. Nos testes acima, os exemplos custam
cerca de 300 tokens extras por pergunta.

## 7. Como reproduzir

De dentro de `backend/`, com o ambiente virtual ativado, o `.env` preenchido
com `GROQ_API_KEY` e o índice FAISS já gerado (`python -m app.ingest`):

```bash
python -m testes.comparar_zero_few_shot
```

```bash
python -m testes.comparar_zero_few_shot --repeticoes 3 --casos factual-simples,conhecimento-proprio
```

```bash
python -m testes.validar_classificacao
```

O último aceita `--http` para testar de ponta a ponta contra `POST /chat` com
o servidor no ar, em vez de chamar a LLM de classificação diretamente.

Os scripts pausam 22s entre chamadas e repetem com backoff quando a Groq
responde 429: o plano gratuito limita 8000 tokens por minuto e cada chamada de
geração consome cerca de 2500.

Saídas geradas:

- `docs/dados/zero-vs-few-shot.json`
- `docs/dados/zero-vs-few-shot-estabilidade.json`
- `docs/dados/validacao-classificacao.json`
