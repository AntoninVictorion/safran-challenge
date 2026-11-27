# Safran Challenge – Détection de panne moteur sur séries temporelles

Projet d'entraînement (maintenance prédictive) : à partir de mesures de capteurs enregistrées cycle après cycle, **détecter tôt la dégradation d'un moteur, limiter les fausses alarmes et pouvoir expliquer l'alerte**.

Trois détecteurs sont construits sur les mêmes données et comparés avec le même protocole : un score de santé simple, XGBoost et un autoencodeur.

## Résultat en bref

| Méthode | Apprentissage | F1 par moteur (test) |
|---|---|---|
| XGBoost | supervisé (étiquette « panne imminente ») | **0,975** |
| Score de santé \|z\| lissé | aucun | 0,941 |
| Autoencodeur | non supervisé (moteurs sains uniquement) | 0,72 à 0,81 selon le seuil |

- Un score de santé simple, causal et sans apprentissage fait presque aussi bien que XGBoost (écart de 2 moteurs sur 259, non significatif), avec une explication directe de chaque alerte.
- Dans cette configuration, l'autoencodeur est nettement moins bon.
- F1 calculé par moteur, au dernier cycle observé, zone de tolérance (RUL 31-60) retirée. Détail et limites plus bas.

## Données

- **NASA C-MAPSS, sous-jeu FD002** : simulation de dégradation de turboréacteurs, 21 capteurs, 3 réglages de fonctionnement qui définissent **6 régimes**, 260 moteurs d'entraînement et 259 moteurs de test.
- Source : NASA Prognostics Data Repository, entrée 6, « Turbofan Engine Degradation Simulation Data Set ».
- Les données ne sont pas versionnées (`data/` est ignoré). Pour reproduire, télécharger l'archive depuis la source ci-dessus et placer `train_FD002.txt`, `test_FD002.txt` et `RUL_FD002.txt` dans `data/`.
- Limites du jeu : un point correspond à un cycle (un vol), sans phases de vol. Il n'y a pas d'étiquette de santé : la durée de vie restante (RUL) se déduit de la fin de chaque trajectoire d'entraînement. Dans le test, les trajectoires s'arrêtent avant la panne et la RUL n'est connue qu'au dernier cycle observé de chaque moteur.

## Méthode

1. **Régimes** : classification des 6 régimes de fonctionnement avec un KMeans sur les trois réglages.
2. **Sélection des capteurs** : 14 capteurs retenus sur 21. Quatre ne font que coder le régime (écart-type nul à régime fixe), deux varient trop peu et un est un simple arrondi.
3. **Normalisation** : z-score par régime, avec la moyenne et l'écart-type des 30 premiers cycles de chaque moteur, supposés sains. Les paramètres sont appris sur le train et réutilisés tels quels sur le test.
4. **Lissage causal** : moyenne glissante sur 10 cycles, passé uniquement, pour que la méthode reste utilisable en vol.
5. **Définition de « panne imminente »** : RUL ≤ 30 cycles. Les cycles de RUL 31 à 60 forment une zone de tolérance, non comptée comme fausse alarme ni comme détection.
6. **Règle d'alerte** : alerte brute puis confirmation par persistance (au moins 3 alertes sur les 5 derniers cycles).

### Les trois détecteurs

- **Score de santé** : moyenne des valeurs absolues des 14 capteurs lissés. Alerte brute si le score dépasse 1,5.
- **XGBoost** : 42 variables (moyenne, écart-type et pente sur 10 cycles pour chacun des 14 capteurs), entraîné pour prédire « imminent ». Alerte brute si la probabilité dépasse 0,5. Découpage entraînement / validation par moteur.
- **Autoencodeur** (PyTorch, 14 → 8 → 4 → 8 → 14) : entraîné uniquement sur les 30 premiers cycles de moteurs sains, sans aucune étiquette de panne. Le score est l'erreur de reconstruction, avec un seuil fixé par percentile (90, 95, 99, 99,5) de l'erreur sur les données saines.

## Résultats

### Validation (52 moteurs, détection par moteur)

| | Détectés | Avance moyenne | Moteurs en fausse alarme |
|---|---|---|---|
| Score de santé | 52/52 | 27,1 cycles | 3 |
| XGBoost | 52/52 | 24,0 cycles | 1 |

Les deux méthodes détectent tous les moteurs. Le score de santé prévient un peu plus tôt, XGBoost déclenche moins de fausses alarmes : c'est un compromis, sans gagnant net.

### Test (259 moteurs jamais vus, alerte au dernier cycle observé)

Part des moteurs en alerte, selon la RUL réelle :

| RUL réelle | Moteurs | Score de santé | XGBoost |
|---|---|---|---|
| ≤ 30 (à détecter) | 61 | 92 % | 95 % |
| 31 – 60 (zone de tolérance) | 41 | 44 % | 17 % |
| 61 – 100 | 65 | 3 % | 0 % |
| > 100 | 92 | 0 % | 0 % |

Zone de tolérance retirée : XGBoost détecte 58 moteurs sur 61 sans aucune fausse alarme (F1 0,975). Le score de santé en détecte 56 avec 2 fausses alarmes (F1 0,941). L'autoencodeur reste en dessous quel que soit le seuil : les fausses alarmes ne diminuent qu'en perdant des détections.

## Explicabilité

Le score de santé se décompose exactement : la contribution d'un capteur est `|z lissé| / 14` et la somme des contributions redonne le score. Le signe du z indique le sens de la dérive.

Sur les 56 moteurs du test détectés à temps :
- `s11`, `s4`, `s15`, `s17`, `s2`, `s3` dérivent à la hausse et `s12`, `s20`, `s7`, `s21` à la baisse **dans 100 % des moteurs**, conformément à l'exploration des données.
- Aucun capteur ne domine l'alerte : les contributions moyennes sont réparties entre 0,11 et 0,28 pour un score moyen d'environ 2,7.
- `s8`, `s9`, `s13`, `s14` contribuent beaucoup mais leur sens de dérive varie d'un moteur à l'autre (positif dans 68 à 84 % des cas) : piste d'amélioration.

## Limites

- Un seul sous-jeu (FD002) et un découpage entraînement / validation avec une seule graine.
- Sur le test, seul le dernier cycle de chaque moteur est jugé : l'avance de détection n'est pas mesurable, et ce critère est plus strict que celui du train.
- L'écart entre XGBoost et le score de santé porte sur 2 moteurs : il n'est pas significatif.
- Le seuil 1,5, la persistance 3/5 et la limite à 60 ont été choisis sur le train. Aucun paramètre n'a été réglé sur le test.
- Autoencodeur : une seule configuration, non optimisée. L'explication proposée (le réseau reconstruit en partie la dérive commune des capteurs) n'est pas vérifiée.
- Les 30 premiers cycles de chaque moteur sont supposés sains : hypothèse non contrôlée. Elle sert à la normalisation et à l'entraînement de l'autoencodeur.

## Reproduire

Le travail est dans le notebook `script.ipynb` (Jupyter, VS Code).

```bash
conda create -n safran python=3.11
conda activate safran
pip install pandas numpy scikit-learn matplotlib jupyter xgboost torch
```

Placer les fichiers de données dans `data/` puis exécuter le notebook de haut en bas. Sous Windows, importer `torch` **avant** pandas, scikit-learn et XGBoost, dans la première cellule d'imports. Un import raté oblige à redémarrer le noyau.

## Pistes d'amélioration

- Comparer le score de santé à 10 capteurs (sans `s8`, `s9`, `s13`, `s14`) sur le test.
- Régler le seuil de XGBoost avec la courbe précision-rappel, tester d'autres graines de découpage, expliquer XGBoost avec SHAP.
- Variantes de l'autoencodeur (goulot plus étroit, erreur par capteur, entrée non lissée).
- Autres détecteurs : Isolation Forest, détection de rupture, estimation de la RUL.
- Dashboard de suivi (Streamlit) et autres sous-jeux (FD001, FD003, FD004).