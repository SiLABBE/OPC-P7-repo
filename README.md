# OPC-P7-repo
 Projet d'implémentation d'un modèle de scoring

Objectif : 
Mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. 
Les 2 principaux axes de ce projet portent donc sur : 
●	Le développement d’un algorithme de classification s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).
●	Le développement et le déploiement d’un dashboard interactif à destination des chargés de relation client permettant d’expliquer de façon la plus transparente possible les décisions d’octroi de crédit.

Découpage des dossiers : 
- Model_training contient le Notebook de développement du modèle, ainsi que celui d'étude du Data Drift (+ tableau Evidently).
- API_Dashboard contient les dossiers API et Dashboard qui intègre les scripts nécessaires au déploiement sur le cloud du modèle et d'un tableau de bord.   