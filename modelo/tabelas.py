#%%
!pip install pandas
!pip install feature-engine
!pip install matplotlib

#%%
import pandas as pd
# %%
tabela_1 = pd.read_parquet("../dados/ParquetCards2024_33.parquet")
# %%
tabela_1.head()
# %%
tabela_2 = pd.read_parquet("../dados/Parquet_Logs2024_33.parquet")
# %%
tabela_2.head()
# %%
tabela_3 = pd.read_parquet("../dados/Parquet_Checklists2024_33.parquet")
# %%
tabela_3.head()
# %%
tabela_4 = pd.read_parquet("../dados/Parquet_itens_Checklists2024_33.parquet")
# %%
tabela_4.head()
# %%

tabela_1.columns
# %%
tabela_1.nunique().sort_values(ascending=False).head(40)
# %%
tabela_1.shape
# %%
tabela_1.isna().sum().sort_values(ascending=False).head(40)
#%%
colunas2 = tabela_2.columns.to_list()
colunas2
# %%
tabela_2.nunique().sort_values(ascending=False).head(40)
# %%
tabela_3.columns

# %%
tabela_4.columns
# %%
tabela_2.shape
# %%

tabela_1.head()

#%%
remover = ['idMembersVoted.@Value', 'idLabels.@Value_u0', 'trello.board',
       'trello.card', 'badges.location', 'badges.votes','badges.fogbugz',
       'cover.idPlugin', 'root.checkItemStates', 'cover.color_u0',
       'root.dueReminder', 'root.idShort', 'root.idAttachmentCover',
       'root.manualCoverAttachment','root.subscribed_u0','root.pos','Código Membro', 'Flag Card Fechado',
       'Flag Pendência Data Card','Descrição Card','Chave Quadro Card','Nome Entrega','Flag Entrega Concluída']
# %%

tabela_1.iloc[:,40:].columns
# %%

tabela_1_limpa = tabela_1.drop(columns=remover)
# %%

tabela_1_limpa.shape
# %%
tabela_1_limpa.head()
# %%
tabela_1_limpa.columns

#%%

tabela_2.iloc[:,80:].columns
# %%
remover2 = ['cover.idAttachment', 'cover.idUploadedBackground', 'cover.size',
       'cover.brightness', 'cover.plugin', 'card.idShort', 'card.shortLink',
       'card.idList', 'cardSource.idShort_u0', 'cardSource.shortLink_u1',
       'board.shortLink_u0', 'cover_u0.idAttachment_u0',
       'cover_u0.idUploadedBackground_u0', 'cover_u0.size_u0',
       'cover_u0.brightness_u0', 'cover_u0.plugin_u0', 'cardSource.id_u6',
       'customFieldItem.id_u3',
       'customFieldItem.idCustomField', 'customFieldItem.idModel',
       'customFieldItem.modelType', 'customFieldItem.value_u0', 'memberCreator.avatarHash',
       'old.value', 'old.idList_u0', 'memberCreator.avatarUrl', 'memberCreator.idMemberReferrer',
       'memberCreator.initials', 'memberCreator.nonPublicAvailable',
       'memberCreator.username', 'member_u0.activityBlocked_u0',
       'member_u0.avatarHash_u0', 'member_u0.avatarUrl_u0',
       'member_u0.idMemberReferrer_u0', 'member_u0.initials_u0',
       'member_u0.nonPublicAvailable_u0', 'member_u0.username_u0',
       'root.appCreator', 'Nome Membro Atribuído Quadro',
       'Código Membro Atribuído Quadro', 'Código Membro Atribuído Card', 'Nome Membro Atribuído Card',
       'Nome Área Trabalho', 'Comentário', 'Nome Anexo', 'Código Anexo',
       'Link Anexo', 'Código Checklist', 'Nome Card Origem',
       'Código Campo Customizado Modificado',
       'Nome Campo Customizado Modificado',
       'Tipo Campo Customizado Modificado', 'Código Item Checklist',
       'Descrição Anterior Card', 'Cor Capa Card',
       ]

#%%

tabela_2_limpa = tabela_2.drop(columns= remover2)
tabela_2_limpa.head()

#%%
tabela_2_limpa.isna().sum().sort_values(ascending = False)
# %%
tabela_2_limpa.shape
# %%

remover3 = ['prefs.isTemplate','boardSource.id_u12','ID Área Trabalho','data.dateLastEdited',
'data.idMemberAdded','data.memberType','data.deactivated','perAction.disableAt',
'perAction.status','uniquePerAction.warnAt_u0','uniquePerAction.disableAt_u0',
'uniquePerAction.status_u0','perAction.warnAt','data.idMember','Código Lista Anterior',
'Código Lista Posterior','Status Item Checklist Log','Nome Lista Inicial','Código Lista Inicial']
#%%

tabela_2_limpa = tabela_2_limpa.drop(columns=remover3).copy()
tabela_2_limpa.head()

#%%

remover4 = tabela_2_limpa.iloc[:,9:-2].columns.to_list()
# %%

tabela_2_limpa['Nome Lista Anterior'].value_counts()
# %%
tabela_2_limpa['Nome Lista Posterior'].value_counts()
# %%
tabela_2_limpa['Posição Anterior Card'].value_counts()
# %%
tabela_2_limpa['Flag Descrição Impedimento'].value_counts()
# %%
remover4
# %%

tabela_2_limpa = tabela_2_limpa.drop(columns = remover4).copy()
tabela_2_limpa.head()
# %%
tabela_2_limpa.shape
# %%
tabela_2_limpa.head()
# %%

tabela_3.head()
# %%

tabela_4.columns
# %%
remover5 = ['Nome Item Checklist','Código Membro','Nome Item Checklist']

tabela_4_limpa = tabela_4.drop(columns=remover5)

tabela_4_limpa.head()

# %%

tabela_master = tabela_1_limpa.copy()
# %%
df_logs = tabela_2_limpa.groupby('Código Card').agg({
    'Código Log':'count',
    'Data Hora Log':'count'
})
# %%
df_logs
# %%
tabela_master = tabela_master.merge(df_logs, on='Código Card', how='left')

tabela_master.head()
#%%
tabela_4_limpa['Status Item Checklist'].value_counts()
# %%
df_check = tabela_4_limpa.groupby('Código Card').agg({
    'Código Item Checklist': 'count',
    'Status Item Checklist': lambda x: (x == 'Completo').sum()
})
df_check
# %%

tabela_master = tabela_master.merge(df_check, on='Código Card', how='left')

tabela_master.head()
# %%

tabela_master.columns
# %%
colunas_finais = ['Código Card','Data Hora Vencimento Card',
'Quantidade Dias Até Vencimento Card','Data Hora Limite Entrega Planejamento',
'Quantidade Dias Até Vencimento Planejamento', 'Flag Vencimento Completo Card']
#%%

tabela_master[colunas_finais].head()
# %%
df_master = tabela_master[colunas_finais].copy()
# %%
df_master.head()
# %%
df_master['Flag Vencimento Completo Card'].value_counts()
# %%
df_master.head()
# %%
df_master = df_master.rename(columns ={
    'Código Card':'Codigo_Card',
    'Data Hora Vencimento Card' : 'Data_Hora_Vencimento_Card',
    'Data Hora Limite Entrega Planejamento':'Data_Hora_Limite_Entrega_Planejamento',
    'Quantidade Dias Até Vencimento Card': 'Quantidade_Dias_Ate_Vencimento_Card',
    'Quantidade Dias Até Vencimento Planejamento' :'Quantidade_Dias_Ate_Vencimento_Planejamento',
    'Flag Vencimento Completo Card' : 'Flag_Vencimento_Completo_Card'
})
# %%
df_master

df_master.to_csv('tabela_mestre.csv')
# %%

df_master.isna().sum()
#%%
df_master = df_master.dropna(subset = ['Data_Hora_Vencimento_Card','Data_Hora_Limite_Entrega_Planejamento'])
df_master.isna().sum()
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