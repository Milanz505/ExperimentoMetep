# Importa as bibliotecas necessárias
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from timeit import default_timer as timer

# Cria uma lista de transações (cada transação é uma lista de itens)
# mushroom.dat é o arquivo texto que contém 8124 transações
transacoes = np.loadtxt("mushroom.dat", dtype=int)

# Converte a lista de transações em um formato adequado para o Apriori
te = TransactionEncoder()
te_ary = te.fit(transacoes).transform(transacoes)
df = pd.DataFrame(te_ary, columns=te.columns_)

#suportes mínimos usados para o teste
SuporteMinimo = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

#tempos médios de execução para os diferentes suportes mínimos
mediaExecucao = []

#Armazena os 10 tempos de execução para um suporte mínimo entre 30% e 60%
temposExecucao30 = []
temposExecucao35 = []
temposExecucao40 = []
temposExecucao45 = []
temposExecucao50 = []
temposExecucao55 = []
temposExecucao60 = []

for value in SuporteMinimo:
    #executa 10 vezes o Apriori
    for i in range(10):
        TempoInicial = timer()
        frequent_itemsets = apriori(df, min_support=value)
        TempoFinal= timer()
        Delta = TempoFinal- TempoInicial
        if value == 0.3: temposExecucao30.append(Delta)
        elif value == 0.35: temposExecucao35.append(Delta)
        elif value == 0.4: temposExecucao40.append(Delta)
        elif value == 0.45: temposExecucao45.append(Delta)
        elif value == 0.5: temposExecucao50.append(Delta)
        elif value == 0.55: temposExecucao55.append(Delta)
        elif value == 0.6: temposExecucao60.append(Delta)

mediaExecucao.append(sum(temposExecucao30)/len(temposExecucao30))
mediaExecucao.append(sum(temposExecucao35)/len(temposExecucao35))
mediaExecucao.append(sum(temposExecucao40)/len(temposExecucao40))
mediaExecucao.append(sum(temposExecucao45)/len(temposExecucao45))
mediaExecucao.append(sum(temposExecucao50)/len(temposExecucao50))
mediaExecucao.append(sum(temposExecucao55)/len(temposExecucao55))
mediaExecucao.append(sum(temposExecucao60)/len(temposExecucao60))

# Plota o gráfico
plt.plot(SuporteMinimo, mediaExecucao, marker='o')
plt.title('Tempo médio de execução do Apriori em relação ao suporte mínimo')
plt.xlabel('Suporte mínimo')
plt.ylabel('Tempo (segundos)')
plt.show()