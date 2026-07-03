# DataTour 2026 — Détection de fraude Mobile Money

Documentation du projet : ce qui a été fait, comment, et pourquoi. Écrite pour être
compréhensible même sans bagage en machine learning.

---

## 1. Le problème, en une phrase

On reçoit des millions de transactions Mobile Money (dépôts, transferts, retraits...)
et on doit dire, pour chacune, **à quel point elle ressemble à une fraude** — pas
juste "oui/non", mais une probabilité entre 0 et 1.

## 2. Pourquoi c'est difficile

- **Très peu de fraudes** par rapport au nombre total de transactions : un modèle
  qui dit "jamais de fraude" aurait déjà raison presque tout le temps, mais serait
  inutile. Il faut un modèle qui sait repérer l'aiguille dans la botte de foin.
- **Les fraudeurs se cachent parmi des comportements qui se ressemblent déjà** :
  la difficulté n'est pas de repérer un type d'opération risqué, mais de distinguer
  la fraude de l'activité normale *à l'intérieur* de ce type d'opération.
- **Les comptes sont anonymisés** (des identifiants sans nom ni visage) : impossible
  d'utiliser "qui" est le client, seulement "comment il se comporte".

## 3. La métrique : comment on juge si le modèle est bon

Le classement de la compétition utilise le **PR-AUC** (aussi appelé *Average
Precision*). Concrètement : on trie toutes les transactions du score le plus
suspect au moins suspect, et on regarde si les vraies fraudes se retrouvent bien
en haut de ce classement. Ce n'est pas la précision brute ("j'ai deviné juste X% du
temps") mais la qualité du **classement par risque** — adapté quand les fraudes
sont rares, car il ne récompense pas un modèle qui se contente de dire "pas
fraude" partout.

## 4. Les données

| Fichier | Rôle |
|---|---|
| `train.csv` | 1,29 million de transactions passées, avec l'étiquette réelle (fraude ou non) |
| `test.csv` | Transactions futures à noter, sans étiquette — c'est ce qu'on doit prédire |
| `submission.csv` | Le fichier qu'on dépose sur la plateforme : un score de 0 à 1 par transaction |

Chaque transaction a : un montant, un compte émetteur et destinataire (avec leurs
soldes avant/après), un type d'opération anonymisé (`op_01` à `op_05`), et une
période (l'équivalent d'un horodatage simplifié).

**Découverte clé, vérifiée dans les données** : sur les 5 types d'opération,
**100% des fraudes se trouvent dans `op_03`** (aucune fraude dans les 4 autres
types). Le pipeline se concentre donc uniquement sur `op_03` — chercher un signal
de fraude dans les autres types serait chercher du bruit.

## 5. La stratégie générale

Le cœur de l'approche repose sur une idée simple : **une transaction n'est pas
suspecte dans l'absolu, elle est suspecte par rapport à l'historique du compte qui
l'effectue**. Un retrait de 500 000 F n'a pas le même sens pour un compte qui en
brasse chaque semaine que pour un compte qui n'a jamais dépassé 10 000 F. Le
pipeline construit donc, pour chaque compte, un "profil comportemental" à partir de
son passé, puis compare chaque nouvelle transaction à ce profil.

Trois familles de signaux sont construites (détail en section 6) :

1. **Le compte est-il cohérent avec sa propre histoire ?** (montants, soldes,
   fréquence habituels)
2. **Le compte est-il actif de façon inhabituelle en ce moment ?** (rafale
   récente de transactions — les fraudeurs agissent souvent vite avant d'être
   bloqués)
3. **Le compte destinataire a-t-il déjà été impliqué dans des fraudes
   passées ?** (repérage des "comptes mules" réutilisés pour collecter l'argent
   volé)

Ces signaux sont ensuite donnés à un modèle de machine learning (des arbres de
décision, voir section 7) qui apprend tout seul à les combiner pour sortir un
score de risque.

## 6. Comment le modèle "regarde" une transaction (feature engineering)

Le fichier [`solve.py`](solve.py) calcule environ 140 variables dérivées par
transaction. En résumé, les familles principales :

- **Cohérence comptable** : est-ce que solde avant − solde après = montant, comme
  attendu ? Un écart est un signal d'anomalie brute.
- **Écart à la normale du compte** : le montant de cette transaction est-il
  beaucoup plus gros que ce que ce compte fait d'habitude (z-score, comparaison au
  maximum historique, etc.) ?
- **Nouveauté** : ce compte, ou cette paire (émetteur → destinataire), a-t-il déjà
  été vu avant ? Les comptes "tout neufs" ou les paires inédites sont plus
  suspects.
- **Vélocité (vitesse d'activité)** : combien de transactions ce compte a-t-il
  faites sur les 5, 10 et 20 dernières périodes ? Une accélération brutale récente
  ("burst") est un indice classique de fraude en cours.
- **Signal "compte mule"** : un compte destinataire qui reçoit beaucoup et
  renvoie beaucoup ailleurs ressemble à un relais utilisé pour blanchir l'argent
  reçu frauduleusement.
- **Taux de fraude historique du destinataire** (`dest_fraud_rate_op03`) — *la
  variable la plus importante du modèle final* : si un compte a déjà reçu de
  l'argent frauduleux par le passé, il est statistiquement bien plus susceptible
  d'en recevoir encore (mule réutilisée). Ce taux est calculé avec un lissage
  statistique (voir encadré ci-dessous) pour ne pas se laisser abuser par un
  compte n'ayant qu'une seule transaction dans son historique.

> **Lissage bayésien, en clair** : si un compte n'a reçu qu'une seule transaction
> et qu'elle était frauduleuse, son "taux de fraude brut" est de 100%. C'est une
> statistique bâtie sur un échantillon d'un seul élément — pas fiable. Le lissage
> ramène ce taux vers la moyenne générale tant qu'on n'a pas assez d'historique
> pour lui faire confiance, un peu comme une note Google avec 1 seul avis compte
> moins qu'une note avec 500 avis.

## 7. Le modèle de machine learning

Deux familles de modèles à arbres de décision sont utilisées en parallèle puis
combinées (moyenne pondérée) :

- **LightGBM**
- **XGBoost**

Ce sont des modèles qui construisent des centaines de petits arbres de décision
successifs, chacun corrigeant les erreurs du précédent. Ils sont bien adaptés à ce
genre de données tabulaires (colonnes de chiffres/catégories), rapides, et gèrent
naturellement le déséquilibre fraude/non-fraude via un paramètre de pondération
(`scale_pos_weight`).

Utiliser **deux modèles différents et les moyenner** réduit le risque qu'une
faiblesse propre à l'un des deux algorithmes pénalise le score final — c'est une
forme de "sagesse des foules" appliquée à des modèles plutôt qu'à des humains.

## 8. Comment on vérifie que ça marche vraiment (validation)

Piège classique en machine learning : un modèle peut sembler excellent sur les
données qu'il a vues à l'entraînement, mais être mauvais sur des données
nouvelles. Il faut donc toujours le tester sur des données qu'il n'a **jamais**
vues.

Ici, comme les données ont une dimension temporelle (des "périodes"), on ne peut
pas juste piocher des lignes au hasard pour construire un ensemble de test — ce
serait tricher, un peu comme réviser un examen avec le corrigé sous les yeux. Le
pipeline découpe donc le temps en trois tranches : on entraîne sur les périodes
anciennes et on vérifie sur les périodes plus récentes, exactement comme ce qui
se passera en vrai (le jeu de test officiel est fait de périodes encore plus
récentes que tout l'historique d'entraînement).

## 9. Le diagnostic et les 3 corrections apportées (juillet 2026)

En reprenant le pipeline existant (version "v22"), un diagnostic complet a été
fait pour vérifier s'il y avait du **surapprentissage** (le modèle "apprend par
cœur" des détails qui ne se reproduiront jamais, au lieu d'apprendre des règles
générales — comme un élève qui mémorise les réponses d'un exercice précis sans
comprendre la méthode, et qui échoue dès que l'énoncé change un peu).

**Constat** : sur les données jamais vues, le modèle notait 0.86 sur les données
d'entraînement mais seulement 0.36 sur les données de test interne — un écart
énorme, signe clair de par-cœur plutôt que de compréhension.

Trois corrections ont été appliquées, chacune vérifiée séparément avant de passer
à la suivante :

### Correction 1 — Arrêt automatique de l'entraînement (*early stopping*)
Avant, le modèle construisait toujours 500 à 600 arbres, sans savoir si c'était le
bon nombre. Désormais, on lui réserve une petite portion de données "test" pendant
l'entraînement, et il s'arrête automatiquement dès qu'ajouter des arbres
supplémentaires n'aide plus à généraliser. Résultat : le nombre d'arbres réel
choisi varie entre 9 et 200 selon les cas — bien moins qu'avant — pour un score
équivalent, en un temps d'entraînement bien plus court.

### Correction 2 — La feature manquante, enfin codée
Le changelog du pipeline mentionnait depuis longtemps une variable de "taux de
fraude par destination" comme nouveauté, mais elle n'avait en réalité **jamais été
programmée** — un oubli. Une fois réellement implémentée (avec le lissage
statistique décrit en section 6), elle est devenue **la variable la plus
importante du modèle**, et le score de validation est passé de 0.360 à 0.361.

### Correction 3 — Cohérence entre l'entraînement et la vraie vie
Problème plus subtil : quand on calcule "la moyenne des montants de ce compte",
le calcul incluait par erreur la transaction elle-même dans sa propre moyenne
pendant l'entraînement — un peu comme calculer la moyenne de classe d'un élève en
comptant deux fois sa propre note. Ça ne peut pas arriver en vrai usage (on ne
connaît jamais à l'avance le résultat qu'on cherche à prédire), donc ce biais
créait une différence artificielle entre ce que le modèle voyait à l'entraînement
et ce qu'il verra en production. La correction retire la contribution de chaque
ligne à ses propres statistiques. Le score de validation n'a pas bougé (ce biais
ne coûtait donc pas de points), mais l'incohérence est éliminée sans aucun coût —
un gain de robustesse "gratuit".

## 10. Historique des scores

| Version | Changement | Score de validation interne |
|---|---|---|
| v17 → v22 | évolutions successives de features | 0.312 → 0.352 (puis 0.351 sur la plateforme) |
| v23 | arrêt automatique de l'entraînement | 0.360 |
| v24 | taux de fraude par destination | **0.3611** |
| v25 | cohérence entraînement/production | 0.3611 (inchangé, mais plus robuste) |

*(Le score interne est historiquement environ 0.01 au-dessus du score réel
constaté sur la plateforme — à confirmer par une soumission.)*

## 11. Comment relancer le pipeline

```bash
.venv/bin/python solve.py
```

Le script charge les données, calcule les variables, valide sur 3 tranches
temporelles, et — seulement si le score de validation dépasse un seuil de
sécurité (`PROXY_THRESHOLD` dans `solve.py`) — régénère `submission.csv`. Si le
score est en dessous du seuil, l'ancien `submission.csv` (déjà connu et fiable)
est conservé pour ne jamais écraser une bonne soumission par une moins bonne.

## 12. Pistes pour la suite

- Recalibrer `PROXY_THRESHOLD` (actuellement à la limite du bruit de mesure pour
  le nouveau pipeline).
- Étendre la correction de cohérence (section 9, correction 3) aux variables de
  vélocité et de diversité réseau, qui n'en bénéficient pas encore.
- Confirmer le score réel en soumettant `submission.csv` sur la plateforme de
  compétition.
