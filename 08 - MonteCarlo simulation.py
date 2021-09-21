# -*- coding: utf-8 -*-
"""
Primeiros estudos de simulação Monte Carlo

Agosto 2021 - Marco Mialaret Júnior
"""
#gerando uma função exponencial por meio de simulação Monte Carlo
#Usada quando conseguimos encontrar a função inversa da CDF
def exponentialRV(rate,n):
    import numpy as np
    #criando uma lista vazia
    exponential_RVs=[]
    for i in range(n):
        exponential_RV = -(1/rate)*np.log(1-np.random.uniform(0,1))
        exponential_RVs.append(exponential_RV)
    return(exponential_RVs)
#uso: 1-criar uma variável com os valores gerados pela função acima. 2-plotar. 
#obs - quanto maior for o conjunto mais próximo estaremos da distribuição teorica.


#gerando distribuição normal padrão quadrada
#usando a ppf
def std_norm_sq_RV(n):
    import numpy as np
    from scipy import stats
    
    std_norm_sq_RVs = []
    for i in range(n):
        std_norm_sq_RV = stats.norm.ppf(np.random.uniform(0,1))**2
        std_norm_sq_RVs.append(std_norm_sq_RV)
    return(std_norm_sq_RVs)

#gerando uma distribuição arbitrária
def gen_special_RVs(n):
    import numpy as np
    from scipy import stats
    special_RVs=[]
    for i in range(n):
        special_RV = (stats.norm.ppf(np.random.uniform(0,1))**2+
                      stats.expon.ppf(np.random.uniform(0,1), scale = 0.5)+
                      stats.norm.ppf(np.random.uniform(0,1)))
        special_RVs.append(special_RV)
    return(special_RVs)

#simulação dos retornos sobre o número de períodos de uma distribuição
#número de universos hipotéticos
def positive_returns(num_sims,num_periods):
    #um primeiro teste: num_sims =5000 e num_periods=1000
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    final_returns = []
    
    for sim_num in range(num_sims):
        time = [0]
        returns = [0]
        for period in range(1, num_periods + 1):
            time.append(period)
            returns.append(returns[period - 1] + stats.laplace.rvs(loc = 0.05,
                                                               scale = 0.07,
                                                               size =1))
        final_returns.append(float(returns[num_periods - 1]))
        plt.plot(time, returns)
    sns.displot(final_returns)

