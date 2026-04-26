#%%
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install feature_engine
!pip install xgboost

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df_master = pd.read_csv('../dados/tabela_mestre.csv')


#%%
df_master.head()
#%%
df_master['Data_Hora_Vencimento_Card'] = pd.to_datetime(df_master['Data_Hora_Vencimento_Card'], format="ISO8601")
df_master['Data_Hora_Limite_Entrega_Planejamento'] = pd.to_datetime(df_master['Data_Hora_Limite_Entrega_Planejamento'], format='ISO8601')
df_master['Data_Hora_Vencimento_Card'] = pd.to_datetime(df_master['Data_Hora_Vencimento_Card'], format= 'ISO8601')
#%%
df_master = df_master.dropna(subset = ['Data_Hora_Vencimento_Card','Data_Hora_Limite_Entrega_Planejamento'])
df_master.isna().sum()
#%%
df_master.info()
#%%
df_master['Dia_do_Vencimento_card'] = df_master['Data_Hora_Vencimento_Card'].dt.dayofweek
df_master['Dia_limite_entrega_planejammento'] = df_master['Data_Hora_Limite_Entrega_Planejamento'].dt.dayofweek
df_master['Vencimento_card_fim_semana'] = (df_master['Dia_do_Vencimento_card'].isin([0,6])).astype(int)
# %%
df_master.isna().sum()
# %%
df_master['Atrasou'] = (df_master['Flag_Vencimento_Completo_Card'].replace({
    'SIM': 0,   # não atrasou
    'NÃO': 1    # atrasou
})).astype(int)
# %%
df_master.head()
# %%
df_master.columns
#%%
df_master2 = df_master[['Codigo_Card',
       'Quantidade_Dias_Ate_Vencimento_Card',
       'Quantidade_Dias_Ate_Vencimento_Planejamento',
       'Atrasou', 'Dia_do_Vencimento_card',
       'Dia_limite_entrega_planejammento','Vencimento_card_fim_semana']]
# %%
df_master2.head()
# %%

target = 'Atrasou'
features = ['Quantidade_Dias_Ate_Vencimento_Card',
       'Quantidade_Dias_Ate_Vencimento_Planejamento', 'Dia_do_Vencimento_card',
       'Dia_limite_entrega_planejammento','Vencimento_card_fim_semana']

# %%
from sklearn.model_selection import train_test_split
#%%
X = df_master2[features]
y = df_master2[target]
#%%
X.isna().sum()
#%%
y.isna().sum()
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
#%%
print(f'Proporcao no treino {y_train.mean()}')
print(f'Proporcao no teste {y_test.mean()}')

#%%
#Discretizacao de variaveis
from feature_engine.discretisation import DecisionTreeDiscretiser
categorizar = ['Dia_limite_entrega_planejammento','Dia_do_Vencimento_card']
variaveis_discre = ['Quantidade_Dias_Ate_Vencimento_Planejamento','Quantidade_Dias_Ate_Vencimento_Card']
tree_discretizaion = DecisionTreeDiscretiser(
    variables = variaveis_discre,
    cv = 3,
    bin_output= 'bin_number',
    regression= False,
    )
X_train = tree_discretizaion.fit_transform(X_train, y_train)
#%% 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


#%%
clf = tree.DecisionTreeClassifier(
    class_weight='balanced',
    max_depth = 5,
    random_state = 42
)
clf.fit(X_train,y_train)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12), dpi = 300)
tree.plot_tree(
    clf,
    max_depth=5,
    filled= True,
    feature_names= X_train.columns,
    class_names= clf.classes_.astype(str)
)
# %%

from sklearn.metrics import classification_report ,accuracy_score, auc, precision_score, roc_curve, roc_auc_score, precision_recall_curve, recall_score

#%%

#Decision Tree Classifier
pred_arvore = clf.predict(X_train)
pred_arvore_teste = clf.predict(X_test)
y_train_prob = clf.predict_proba(X_train)[:,1]
#%%
precision_teste = precision_score(y_test,pred_arvore_teste)
recall_teste = recall_score(y_test, pred_arvore_teste)
print(classification_report(y_train, pred_arvore))
print(f'Precisao  em teste {precision_teste}')
print(f'Recall em teste {recall_teste}')
# %%

#Random Forest Classifier
random_forest = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None,
    min_samples_leaf =20,
    random_state=42,
    class_weight= 'balanced'
)

random_forest.fit(X_train,y_train)


pred_random = random_forest.predict(X_train)
pred_random_test = random_forest.predict(X_test)
y_train_prob_random = random_forest.predict_proba(X_train)[:,1]
precision_treino = precision_score(y_train,pred_random)
precision_teste = precision_score(y_test, pred_random_test )
recall_teste = recall_score(y_test, pred_random_test)
print(classification_report(y_train, pred_random))
print(f'Precisao  em treino  {precision_treino}')
print(f'Precisao em Teste {precision_teste}')
print(f'Recall em teste {recall_teste}')
#%%
#Logistic Regression

from sklearn.linear_model import LogisticRegression


log = LogisticRegression(
    class_weight= 'balanced'
)

log.fit(X_train, y_train)

log_pred = log.predict(X_train)
log_pred_test = log.predict(X_test)
y_train_prob_log = log.predict_proba(X_train)[:,1]

precision_treino_log = precision_score(y_train,log_pred)
precision_test_log = precision_score(y_test, log_pred_test)
recall_teste = precision_recall_curve(y_test, log_pred_test)
print(classification_report(y_train, log_pred))
print(f'Precisao  em treino {precision_treino_log}')
print(f'Precisao em teste {precision_test_log}')
print(f'Recall em teste {recall_teste}')

#%%
#XGBClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
#%%

bst = XGBClassifier(
    n_estimators =3,
    max_depth= 5,
    learning_rate = 1,
    objective='binary:logistic'
)

bst.fit(X_train, y_train)

bst_pred = bst.predict(X_train)
bst_pred_teste = bst.predict(X_test)

bst_precision_train = precision_score(y_train, bst_pred)
bst_precision_test = precision_score(y_test, bst_pred_teste)
bst_recall = recall_score(y_test, bst_pred_teste)
print(classification_report(y_train, bst_pred))
print(f'Precisao em treino {bst_precision_train}')
print(f'Precisao em teste {bst_precision_test}')
print(f'Recall em Teste {bst_recall}')
#%%

params = {
    'n_estimators': [200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.03, 0.05, 0.1],
    'scale_pos_weight': [5, 10, 20]
}
grid = GridSearchCV(
    estimator = bst,
    param_grid =  params,
    cv = 5,
    scoring = 'precision'

)

grid.fit(X_train,y_train)
#%%


grid_pred = grid.predict(X_train)
grid_pred_teste = grid.predict(X_test)

grid_precision_train = precision_score(y_train, bst_pred)
grid_precision_test = precision_score(y_test, bst_pred_teste)
grid_recall = recall_score(y_test, bst_pred_teste)
print(classification_report(y_train, grid_pred))
print(f'Precisao em treino {grid_precision_train}')
print(f'Precisao em teste {grid_precision_test}')
print(f'Recall em Teste {grid_recall}')



#%%
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6), dpi=300)

modelos = {
    "Decision Tree": clf,
    "Random Forest": random_forest,
    "Regressao Logistica": log,
    'XGBost' : bst,
    'GridSearchCV' : grid
}

for nome, modelo in modelos.items():
    
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Comparação de Modelos")
plt.legend()
plt.show()


#Nova análise
#%%
#Criando uma base de dados balanceada

df_master2 = df_master2.sample(frac=1)

atraso_df = df_master2.loc[df_master2['Atrasou'] == 1]
natraso_df = df_master2.loc[df_master2['Atrasou'] == 0][:124]

normal_distributed_df = pd.concat([atraso_df, natraso_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

#%%
df_master2['Atrasou'].sum()
# %%
numerics = ['Quantidade_Dias_Ate_Vencimento_Card','Quantidade_Dias_Ate_Vencimento_Planejamento','Atrasou','Dia_do_Vencimento_card','Dia_limite_entrega_planejammento']
#%%
corr = df_master2[numerics].corr()
corr_balanceado = new_df[numerics].corr()
# %%

!pip install matplotlib
import matplotlib.pyplot as plt


#%%
f, axes = plt.subplots(ncols=2, figsize=(20,4))

sns.heatmap(
    corr,
    ax= axes[0],

)
axes[0].set_title('Grafico de correlacao desbalanceado')
sns.heatmap(
    corr_balanceado,
    ax= axes[1] 
)
axes[1].set_title('Grafico de correlacao balanceado')
# %%

f, axes = plt.subplots(ncols=2, figsize=(20,4))
sns.boxplot(
    x = 'Atrasou', y = 'Quantidade_Dias_Ate_Vencimento_Planejamento', data = df_master2, ax = axes[0]
)
axes[0].set_title('Quantidade de dias ate o Vencimento do Planejamento')

sns.boxplot(
    x = 'Atrasou', y = 'Quantidade_Dias_Ate_Vencimento_Card', data = df_master2, ax = axes[1]
)
axes[1].set_title('Quantidade de dias ate o Vencimento do Card')

# %%

f, axes = plt.subplots(ncols=2, figsize=(20,4))
sns.boxplot(
    x = 'Atrasou', y = 'Quantidade_Dias_Ate_Vencimento_Planejamento', data = new_df, ax = axes[0]
)
axes[0].set_title('Quantidade de dias ate o Vencimento do Planejamento')

sns.boxplot(
    x = 'Atrasou', y = 'Quantidade_Dias_Ate_Vencimento_Card', data = new_df, ax = axes[1]
)
axes[1].set_title('Quantidade de dias ate o Vencimento do Card')

# %%

X2 = new_df[features]
y2 = new_df[target]

#%%
X2.shape
#%%
y2.shape
#%%
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size= 0.2, random_state=42, stratify= y2)
#%%
print(f'Proporcao no treino {y_train2.mean()}')
print(f'Proporcao no teste {y_test2.mean()}')
#%%
#Discretizacao de variaveis

X_train2 = tree_discretizaion.fit_transform(X_train2, y_train2)

#%% 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#%%
random_forest = RandomForestClassifier(
    n_estimators=300, 
    max_depth=None,
    min_samples_leaf =15,
    random_state=42,

)

#%%
clf = tree.DecisionTreeClassifier(
    max_depth = 5,
    random_state = 42
)
clf.fit(X_train2,y_train2)
#%%
random_forest.fit(X_train2,y_train2)
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12), dpi = 300)
tree.plot_tree(
    clf,
    max_depth=5,
    filled= True,
    feature_names= X_train2.columns,
    class_names= clf.classes_.astype(str)
)
# %%

from sklearn.metrics import classification_report , accuracy_score ,auc, precision_score, roc_curve, roc_auc_score, precision_recall_curve

#%%

#Decision Tree Classifier
pred_arvore2 = clf.predict(X_train2)
pred_arvore_teste2 = clf.predict(X_test2)
pred_arvore_treino = clf.predict(X_train)
y_train_prob2= clf.predict_proba(X_train2)[:,1]
#%%
accuray_treino = accuracy_score(y_train2, pred_arvore2)
precision_treino = precision_score(y_train2,pred_arvore2)
precision_teste2 = precision_score(y_test2, pred_arvore_teste2)
precision_teste = precision_score(y_train, pred_arvore_treino)
print(classification_report(y_train2, pred_arvore2))
print(f'Precisao  em treino {precision_treino}')
print(f'Precisao em teste balanceado {precision_teste2}')
print(f'Precisao em teste real {precision_teste}')
print(44*'-')

print(classification_report(y_train, pred_arvore_treino))
print(f'Precisao em teste {precision_teste}')

# %%

#Random Forest Classifier
pred_random_balanceado = random_forest.predict(X_train2)
pred_random_real = random_forest.predict(X_test)
pred_random_test_balanceado = random_forest.predict(X_test2)
#y_train_prob_random = random_forest.predict_proba(X_train2)[:,1]
accuray_treino_random = accuracy_score(y_train2, pred_random_balanceado)
precision_treino_balanceado = precision_score(y_train2,pred_random_balanceado)
precision_teste_balanceado = precision_score(y_test2, pred_random_test_balanceado )
precision_teste_real = precision_score(y_test, pred_random_real)
#%%

print(classification_report(y_train2, pred_random_balanceado))
print(f'Precisao  em treino balanceado {precision_treino_balanceado}')
print(f'Precisao em Teste balanceado {precision_teste_balanceado}')
print(f'Precisao em Teste Real {precision_teste_real}')
#%%
print(classification_report(y_test, pred_random_real))

#%%
#Logistic Regression

from sklearn.linear_model import LogisticRegression


log = LogisticRegression(
    class_weight= 'balanced'
)

log.fit(X_train, y_train)

log_pred = log.predict(X_train2)
log_pred_test = log.predict(X_test2)
y_train_prob_log = log.predict_proba(X_train2)[:,1]

precision_treino_log = precision_score(y_train2,log_pred)
precision_test_log = precision_score(y_test2, log_pred_test)
print(classification_report(y_train2, log_pred))
print(f'Precisao  em treino {precision_treino_log}')
print(f'Precisao em teste {precision_test_log}')




#%%
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6), dpi=300)

modelos = {
    "Decision Tree": clf,
    "Random Forest": random_forest,
    "Regressao Logistica": log
}

for nome, modelo in modelos.items():
    
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Comparação de Modelos")
plt.legend()
plt.show()

# %%
