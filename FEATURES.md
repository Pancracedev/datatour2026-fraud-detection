# Dictionnaire des variables (features) du modèle

Ce document liste **toutes les variables** que le modèle utilise réellement pour
noter une transaction, calculées par `compute_stats()` et `build_feats()` dans
[`solve.py`](solve.py). Il complète la vue d'ensemble du [README](README.md)
(section 6) avec le détail complet, variable par variable.

**Comment lire ce document** : chaque variable est un nombre que le modèle reçoit
en entrée, en plus de toutes les autres, pour une transaction donnée. Le modèle ne
"comprend" pas leur sens — c'est à nous de construire des variables qui, une fois
combinées, révèlent un comportement suspect. La colonne *Pourquoi* explique le
raisonnement métier derrière chaque variable.

> Convention de nommage : `orig_` = compte émetteur, `dest_` = compte
> destinataire, `_log` = version "compressée" d'un nombre qui peut être très
> grand (voir encadré en fin de document), `_z` = écart à la moyenne exprimé en
> nombre d'écarts-types ("z-score").

---

## 1. Champs bruts de la transaction

Utilisés tels quels, en plus de toutes les variables dérivées ci-dessous.

| Variable | Ce qu'elle mesure |
|---|---|
| `amount` | Montant de la transaction |
| `origin_balance_before` / `origin_balance_after` | Solde du compte émetteur avant/après |
| `destination_balance_before` / `destination_balance_after` | Solde du compte destinataire avant/après |

## 2. Cohérence comptable de la transaction elle-même

Ces variables ne regardent que la transaction en cours, sans historique — un peu
comme vérifier qu'une facture s'additionne correctement.

| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `mismatch_orig` / `mismatch_orig_log` | Écart entre (solde avant − solde après) et le montant déclaré, côté émetteur | Un écart signale une transaction mal formée ou une manipulation |
| `mismatch_dest` | Même écart côté destinataire | Idem |
| `orig_neg_after` | Le compte émetteur passe-t-il en négatif après la transaction ? | Un compte drainé au-delà de ses moyens est un signal fort |
| `orig_zero_after` | Le compte émetteur tombe-t-il à zéro ? | Vidage complet du compte — typique d'une fraude en cours |
| `orig_drained` | Le compte émetteur perd-il quasiment tout son solde (moins de 1% restant) ? | Version plus souple de "compte vidé" |
| `dest_zero_before` | Le compte destinataire était-il vide avant de recevoir ? | Un compte "coquille vide" qui s'active soudainement est suspect |
| `dest_zero_after` | Le compte destinataire est-il toujours quasi vide juste après avoir reçu ? | Suggère que l'argent reçu est aussitôt retransféré ailleurs |
| `dest_bal_decrease` | Le solde du destinataire baisse-t-il alors qu'il reçoit de l'argent ? | Anomalie comptable à surveiller |
| `amount_log` | Montant, en échelle compressée | Évite qu'une transaction énorme n'écrase le signal des autres |
| `orig_bal_log` / `dest_bal_log` | Soldes, en échelle compressée | Idem |
| `amt_ratio_orig` | Montant ÷ solde du compte émetteur | Une transaction qui représente 100% du solde est plus risquée qu'une qui en représente 1% |
| `amt_ratio_dest` | Montant ÷ solde du compte destinataire | Idem, vu du côté récepteur |
| `exceeds_orig` | Le montant dépasse-t-il le solde disponible ? | Signal d'anomalie brut |
| `net_flow_norm` | Variation nette des deux soldes, rapportée au montant | Détecte les mouvements d'argent qui ne "collent" pas avec la transaction déclarée |
| `balance_ratio` | Solde du destinataire ÷ solde de l'émetteur | Compare la "taille" des deux comptes impliqués |

## 3. Le compte est-il nouveau ou connu ?

| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `orig_is_new` | Le compte émetteur n'a jamais été vu dans l'historique | Un compte tout juste créé qui envoie déjà de l'argent est plus risqué qu'un compte ancien |
| `dest_is_new` | Le compte destinataire n'a jamais été vu | Idem, côté réception — souvent le profil d'une mule fraîchement recrutée |
| `orig_no_clean_hist` | Le compte émetteur n'a **aucun** historique "propre" (hors op_03) | On n'a aucune référence de comportement normal pour ce compte |
| `dest_no_clean_hist` | Idem, côté destinataire | — |

## 4. Profil comportemental "propre" du compte (hors transactions op_03)

Calculé uniquement à partir des opérations *autres que* `op_03` (des opérations où
il n'y a jamais de fraude) — c'est la référence de "comportement normal" d'un
compte, indépendante de toute transaction potentiellement frauduleuse.

| Variable | Ce qu'elle mesure |
|---|---|
| `orig_clean_n` | Nombre de transactions "propres" déjà faites par ce compte émetteur |
| `orig_clean_mean` / `orig_clean_std` | Montant moyen et variabilité habituels de ce compte |
| `orig_clean_max` / `orig_clean_q75` | Montant maximum et "haut de fourchette" (75ᵉ percentile) habituels |
| `orig_clean_bal_mean` / `orig_clean_bal_std` | Solde moyen et sa variabilité pour ce compte |
| `dest_clean_n` / `dest_clean_mean` / `dest_clean_std` | Équivalents côté compte destinataire |
| `dest_clean_bal_mean` | Solde moyen habituel du destinataire |

## 5. Statistiques toutes opérations confondues

Comme la section 4, mais calculées sur **toutes** les opérations (y compris
`op_03`) — donne une vue plus large de l'activité globale du compte.

| Variable | Ce qu'elle mesure |
|---|---|
| `orig_all_n` / `orig_all_mean` / `orig_all_std` | Nombre, montant moyen, variabilité — toutes opérations, compte émetteur |
| `orig_tenure` | Ancienneté du compte (écart entre sa 1ʳᵉ et sa dernière période observée) |
| `dest_all_n` / `dest_all_mean` / `dest_all_std` | Équivalents, compte destinataire |
| `dest_all_bal_mean` / `dest_all_bal_std` | Solde moyen et variabilité du destinataire |
| `orig_all_sum`, `orig_all_sumsq`, `dest_all_sum`, `dest_all_sumsq`, `dest_all_bal_sum`, `dest_all_bal_sumsq` | Valeurs intermédiaires (somme, somme des carrés) qui servent à recalculer moyenne/écart-type sans la transaction en cours (voir section 12, *leave-one-out*) — pas conçues pour être lues directement, mais bien fournies au modèle |

## 6. Diversité du réseau de contacts

| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `orig_n_unique_dests` | À combien de destinataires *différents* ce compte a-t-il déjà envoyé de l'argent ? | Un compte qui envoie toujours au même destinataire a un profil différent d'un compte qui disperse ses envois |
| `dest_n_unique_origs` | Combien d'émetteurs *différents* ont déjà envoyé à ce destinataire ? | Un compte qui collecte l'argent de nombreuses sources différentes ressemble à un point de collecte |
| `dest_n_unique_origs_op03` | Idem, mais en ne comptant que les transactions `op_03` | Version ciblée sur le type d'opération à risque |

## 7. Statistiques spécifiques à `op_03` (le type d'opération à risque)

| Variable | Ce qu'elle mesure |
|---|---|
| `orig_op03_n` / `orig_op03_mean` / `orig_op03_std` | Nombre, montant moyen, variabilité des transactions `op_03` de ce compte émetteur |
| `orig_op03_max` / `orig_op03_q75` | Montant maximum et haut de fourchette habituels sur `op_03` |
| `orig_op03_active_periods` | Sur combien de périodes différentes ce compte a-t-il été actif en `op_03` ? |
| `dest_op03_n` / `dest_op03_mean` / `dest_op03_std` | Équivalents côté destinataire |
| `dest_op03_bal_mean` | Solde moyen du destinataire lors de ses réceptions `op_03` |
| `orig_op03_sum`, `orig_op03_sumsq`, `dest_op03_sum`, `dest_op03_sumsq` | Valeurs intermédiaires pour le calcul leave-one-out (section 12) |

## 8. Signal "compte mule"

Repère les comptes qui font office de relais : ils reçoivent de l'argent puis le
redistribuent ailleurs, un schéma classique de blanchiment.

| Variable | Ce qu'elle mesure |
|---|---|
| `dest_as_orig_n` / `dest_as_orig_sum` | Nombre et montant total des transactions où ce compte destinataire agit lui-même comme émetteur ailleurs |
| `dest_as_orig_log` | Version compressée de `dest_as_orig_n` |
| `dest_mule_ratio` | Part de l'activité de ce compte qui consiste à *renvoyer* de l'argent plutôt qu'à en recevoir passivement |
| `dest_as_orig_sum_log` | Version compressée du volume renvoyé |
| `dest_relay_score` | Score combiné : ce compte reçoit-il beaucoup **et** renvoie-t-il beaucoup ? (relais actif) |

## 9. Vélocité — activité récente (3 fenêtres temporelles)

Un fraudeur agit souvent vite, en rafale, avant que son compte ne soit bloqué. Ces
variables comparent l'activité **récente** d'un compte à son activité habituelle,
sur trois échelles de temps.

### Fenêtre longue (20 dernières périodes)
| Variable | Ce qu'elle mesure |
|---|---|
| `orig_vel_n` / `orig_vel_amt_sum` | Nombre et montant total des transactions récentes (toutes opérations) du compte émetteur |
| `orig_vel_n_ops` | Nombre de types d'opérations différents utilisés récemment |
| `orig_vel_op03_n` / `orig_vel_op03_sum` | Idem, limité aux transactions `op_03` |
| `dest_vel_n` / `dest_vel_amt_sum` | Équivalents côté destinataire (toutes opérations) |
| `dest_vel_op03_n` | Nombre de réceptions `op_03` récentes du destinataire |
| `orig_vel_ratio` / `dest_vel_ratio` | Part de l'activité *totale* du compte qui s'est produite récemment (proche de 1 = compte anormalement concentré sur la période récente) |
| `orig_vel_active` / `dest_vel_active` | Le compte a-t-il eu au moins une activité `op_03` récente ? |
| `orig_vel_op03_log`, `dest_vel_op03_log`, `orig_vel_amt_log` | Versions compressées des variables ci-dessus |

### Fenêtre intermédiaire (10 dernières périodes)
| Variable | Ce qu'elle mesure |
|---|---|
| `dest_vel10_amt_sum` / `dest_vel10_op03_n` | Montant total reçu / nombre de réceptions `op_03` sur 10 périodes |
| `dest_vel10_amt_log`, `dest_vel10_op03_log` | Versions compressées |

### Fenêtre courte (5 dernières périodes — le "burst")
| Variable | Ce qu'elle mesure |
|---|---|
| `orig_vel5_op03_n` | Nombre de transactions `op_03` du compte émetteur sur les 5 dernières périodes |
| `dest_vel5_n` / `dest_vel5_amt_sum` | Nombre et montant reçus (toutes opérations) sur 5 périodes |
| `dest_vel5_op03_n` | Nombre de réceptions `op_03` sur 5 périodes |
| `orig_vel5_op03_log`, `dest_vel5_amt_log`, `dest_vel5_op03_log` | Versions compressées |

### Accélération (comparaison entre fenêtres)
| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `orig_burst_accel` | Activité sur 5 périodes comparée au rythme moyen sur 20 périodes (compte émetteur) | Une valeur très supérieure à 1 = rafale récente anormale |
| `dest_burst_accel` | Idem, compte destinataire, en nombre de transactions | — |
| `dest_recv_accel` | Idem, en montant reçu | — |
| `dest_accel_10_20` | Rythme des 10 dernières périodes comparé aux 20 dernières | Détecte une accélération à moyen terme |
| `dest_accel_5_10` | Rythme des 5 dernières périodes comparé aux 10 dernières | Détecte une accélération à très court terme |
| `orig_op03_per_period` | Nombre moyen de transactions `op_03` par période *active* pour ce compte (densité habituelle) | Sert de référence pour juger si l'activité récente est anormalement dense |
| `orig_op03_burst_ratio` | Activité des 5 dernières périodes comparée à cette densité habituelle | Version affinée de `orig_burst_accel`, tenant compte de la fréquence propre du compte |

## 10. Relation entre un émetteur et un destinataire précis (la "paire")

| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `pair_n` | Combien de fois cet émetteur a-t-il déjà envoyé à ce destinataire précis (en `op_03`) ? | Une paire jamais vue avant est plus suspecte qu'une relation habituelle |
| `pair_amt_mean` / `pair_amt_std` | Montant moyen et variabilité habituels de cette paire précise | — |
| `pair_amt_max` | Montant maximum déjà échangé entre ces deux comptes | — |
| `pair_is_new` | Cette paire émetteur→destinataire est-elle inédite ? | — |
| `pair_amt_z` | Écart du montant actuel par rapport à l'habitude de cette paire (en écarts-types) | — |
| `pair_exceeds_max` | Le montant dépasse-t-il tout ce qui a déjà été échangé entre ces deux comptes ? | — |
| `pair_amt_sum`, `pair_amt_sumsq` | Valeurs intermédiaires pour le calcul leave-one-out (section 12) | — |

## 11. Taux de fraude historique du destinataire — *la variable la plus importante*

Ajoutée lors du dernier passage d'amélioration (section 9 du README), c'est la
variable qui pèse le plus dans les décisions du modèle final.

| Variable | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| `dest_fraud_count_op03` | Combien de fraudes confirmées ce compte a-t-il déjà reçues par le passé ? | Un compte déjà impliqué dans des fraudes est un candidat naturel de "mule réutilisée" |
| `dest_fraud_n_op03` | Combien de transactions `op_03` ce compte a-t-il reçues au total ? | Sert de dénominateur pour calculer un taux fiable |
| `dest_fraud_rate_op03` | Taux de fraude historique du destinataire, **lissé statistiquement** pour ne pas se fier à un historique trop court (voir README section 6) | Signal direct et le plus discriminant du modèle |
| `dest_fraud_count_log` | Version compressée de `dest_fraud_count_op03` | — |

## 12. Écarts à la normale (z-scores et ratios dérivés)

Une fois les profils comportementaux ci-dessus rapatriés sur chaque transaction,
ces variables mesurent **l'écart** entre la transaction actuelle et ce profil.

| Variable | Ce qu'elle mesure |
|---|---|
| `amt_z_clean_orig` | Écart du montant par rapport à l'historique "propre" de l'émetteur, en écarts-types |
| `amt_z_all_orig` / `amt_z_all_dest` | Idem, sur l'historique toutes opérations, émetteur / destinataire |
| `amt_z_op03_orig` | Idem, sur l'historique `op_03` de l'émetteur |
| `bal_z_clean_orig` | Écart du solde émetteur par rapport à son solde habituel |
| `amt_pct_clean_max` / `amt_pct_op03_max` | Montant actuel en proportion du maximum jamais vu pour ce compte |
| `amt_vs_q75_clean` / `amt_vs_q75_op03` | Le montant dépasse-t-il le "haut de fourchette" habituel (75ᵉ percentile) ? |
| `orig_vol_clean_log` / `dest_vol_clean_log` | Volume d'activité "propre" du compte, en échelle compressée |
| `orig_vol_op03_log` / `dest_vol_op03_log` | Volume d'activité `op_03` du compte, en échelle compressée |
| `orig_tenure_log` | Ancienneté du compte, en échelle compressée |
| `orig_net_log` / `dest_net_log` | Diversité du réseau de contacts, en échelle compressée |
| `dest_origs_op03_log` | Diversité des émetteurs reçus par ce destinataire (en `op_03`), en échelle compressée |
| `dest_avg_op03_per_orig` | En moyenne, combien de fois chaque émetteur envoie-t-il à ce destinataire ? (proche de 1 = chaque émetteur ne vient qu'une fois, profil de collecte type mule) |

## 13. Variables d'interaction

Combinent deux signaux déjà connus quand leur combinaison est plus parlante que
chaque signal pris isolément.

| Variable | Combine | Pourquoi |
|---|---|---|
| `drain_to_new` | Compte vidé × destinataire inconnu | Vider son compte vers un inconnu est plus suspect que vers un contact habituel |
| `drain_to_newpair` | Compte vidé × relation émetteur-destinataire inédite | — |
| `high_ratio_newdest` | Montant représentant une grosse part du solde × destinataire inconnu | — |
| `exceed_and_newpair` | Montant record pour cette paire × paire inédite | — |
| `bal_z_x_ratio` | Écart de solde × ratio montant/solde | Renforce le signal quand les deux indices pointent dans le même sens |
| `vel_x_drain` | Activité récente élevée × compte vidé | Rafale d'activité qui se termine par un compte vidé |
| `new_dest_vel_burst` | Destinataire inconnu × activité récente élevée de l'émetteur | — |
| `large_to_mule` | Gros montant relatif × profil de compte mule du destinataire | Combine les deux signaux les plus directement liés à la fraude |
| `burst_drain` | Rafale très récente (5 périodes) × compte vidé | Version resserrée de `vel_x_drain` |
| `new_dest_burst5` | Destinataire inconnu × rafale très récente | — |

---

## Note technique : pourquoi des versions "`_log`" ?

Beaucoup de variables ont une version compressée (suffixe `_log`, calculée comme
`log(1 + valeur)`). Sans cette compression, un compte qui a fait 10 000
transactions écraserait complètement, aux yeux du modèle, la différence entre un
compte qui en a fait 5 et un qui en a fait 50 — alors que cette différence-là est
souvent plus significative. La transformation logarithmique rapproche les échelles
pour que le modèle puisse distinguer les nuances aussi bien en bas qu'en haut de
l'échelle.

## Note technique : le "leave-one-out" (section 5, 7, 10)

Certaines colonnes intermédiaires (`*_sum`, `*_sumsq`) ne sont pas des signaux
métier en soi : elles permettent de recalculer une moyenne ou un écart-type **en
retirant la transaction en cours** de son propre calcul, uniquement pendant
l'entraînement (voir README section 9, correction 3). C'est ce qui garantit que le
modèle apprend sur des données qui ressemblent exactement à ce qu'il verra en
production.
