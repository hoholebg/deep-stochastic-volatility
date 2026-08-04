# 🎓 Guide Pédagogique & Didactique : Deep Learning, PINNs & Neural SDEs pour la Finance Quantitative

Bienvenue dans ce guide conçu pour vous permettre de comprendre en profondeur la théorie, les mathématiques et l'implémentation pratique des **Physics-Informed Neural Networks (PINNs)** et des **Neural Stochastic Differential Equations (Neural SDEs)** appliqués au pricing d'options et de produits structurés.

---

## 📚 Table des Matières
1. [Introduction & Intuition : Pourquoi le Deep Learning en Quant Finance ?](#1-introduction--intuition)
2. [L'Équation aux Dérivées Partielles (EDP) de Black-Scholes](#2-léquation-aux-dérivées-partielles-edp-de-black-scholes)
3. [Comment Fonctionne un Physics-Informed Neural Network (PINN) ?](#3-comment-fonctionne-un-physics-informed-neural-network-pinn)
4. [L'Autodifférenciation (`autograd`) et le Calcul des Grecs ($\Delta, \Gamma$)](#4-lautodifférenciation-autograd-et-le-calcul-des-grecs-)
5. [Extensions aux Produits Dépendants de la Trajectoire & Structurés](#5-extensions-aux-produits-dépendants-de-la-trajectoire--structurés)
6. [Analyse Didactique de vos Résultats Numériques ($NVDA$, PINN vs MC vs FDM)](#6-analyse-didactique-de-vos-résultats-numériques)

---

## 1. Introduction & Intuition : Pourquoi le Deep Learning en Quant Finance ?

Traditionnellement, la finance quantitative s'appuie sur deux familles de méthodes numériques pour évaluer les contrats financiers :
- **La Méthode des Différences Finies (FDM)** : Résolution de l'EDP sur une grille spatio-temporelle discrète. *Limite* : Fléau de la dimensionnalité (exponentiel avec le nombre d'actifs sous-jacents).
- **Les Simulations Monte Carlo** : Échantillonnage de trajectoires stochastiques. *Limite* : Lente pour les calculs en temps réel et coûteuse pour le calcul des Grecs.

### 💡 L'Intuition des PINNs
Au lieu d'utiliser une grille rigide ou de tirer 100 000 trajectoires à chaque réévaluation, nous entraînons un **Réseau de Neurones Artificiels** $V_{\theta}(S, t)$ qui apprend à satisfaire **directement l'équation différentielle de la finance**.

Une fois entraîné, le réseau agit comme un **opérateur instantané** : l'évaluation du prix d'un produit financier sur 1 000 sous-jacents différents prend **moins de 1 milliseconde**.

---

## 2. L'Équation aux Dérivées Partielles (EDP) de Black-Scholes

Pour un sous-jacent $S_t$ suivant un Mouvement Brownien Géométrique sous la mesure risque-neutre :
$$dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}$$

Le principe d'arbitrage (portefeuille de couverture parfaite) mène à l'EDP de Black-Scholes :
$$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

En effectuant le changement de variable $\tau = T - t$ (temps restant avant la maturité) :
$$\frac{\partial V}{\partial \tau} = \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

---

## 3. Comment Fonctionne un Physics-Informed Neural Network (PINN) ?

Un PINN est un réseau de neurones standard dont la **fonction de perte (Loss Function)** est contrainte par les lois de la physique ou de la finance.

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{PDE}}(\theta) + \lambda_1 \mathcal{L}_{\text{IC}}(\theta) + \lambda_2 \mathcal{L}_{\text{BC}}(\theta)$$

### 1. La Perte EDP ($\mathcal{L}_{\text{PDE}}$)
Nous échantillonnons des points $(S, \tau)$ dans l'espace. La perte mesure à quel point le réseau viole l'EDP de Black-Scholes :
$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \frac{\partial V_\theta}{\partial \tau} - \left( \frac{1}{2}\sigma^2 S_i^2 \frac{\partial^2 V_\theta}{\partial S^2} + r S_i \frac{\partial V_\theta}{\partial S} - r V_\theta \right) \right|^2$$

### 2. La Condition Initiale ($\mathcal{L}_{\text{IC}}$)
À maturité ($\tau = 0$), le prix de l'option doit égaler exactement le payoff du contrat (ex: $\max(S - K, 0)$ pour un Call) :
$$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{ic}}} \sum_{i=1}^{N_{\text{ic}}} \left| V_\theta(S_i, 0) - \max(S_i - K, 0) \right|^2$$

### 3. Les Conditions aux Limites ($\mathcal{L}_{\text{BC}}$)
- Si $S \to 0$, l'option d'achat ne vaut rien : $V(0, \tau) = 0$.
- Si $S \to \infty$, l'option tend vers la valeur ajustée de l'action : $V(S_{\max}, \tau) = S_{\max} - K e^{-r\tau}$.

---

## 4. L'Autodifférenciation (`autograd`) et le Calcul des Grecs ($\Delta, \Gamma$)

Dans les méthodes classiques, calculer le Delta ($\Delta = \frac{\partial V}{\partial S}$) nécessite des réévaluations par différences finies :
$$\Delta \approx \frac{V(S + \epsilon) - V(S - \epsilon)}{2\epsilon}$$

Dans PyTorch, le graphe de calcul du réseau de neurones permet de calculer les **dérivées partielles exactes** via la rétropropagation (`torch.autograd.grad`) :

```python
# Extrait du code src/pinn_solver.py
S.requires_grad_(True)
tau.requires_grad_(True)

V = model(S, tau)

# Delta exact sans approximation numérique !
Delta = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

# Gamma exact (dérivée seconde)
Gamma = torch.autograd.grad(Delta, S, grad_outputs=torch.ones_like(Delta), create_graph=True)[0]
```

---

## 5. Extensions aux Produits Dépendants de la Trajectoire & Structurés

### A. Options Asiatiques (Dépendance de la Moyenne Arithmétique)
Le payoff dépend de la moyenne arithmétique de l'action : $\bar{S} = \frac{1}{N} \sum_{i=1}^N S_{t_i}$.
Le réseau de neurones prend en entrée le vecteur d'état élargi : $(S_t, \bar{S}_t, \tau)$.

### B. Notes Structurées Phoenix Autocall
Les Autocalls combinent :
- **Observation trimestrielle** : Si $S_{t_k} \ge S_0$, remboursement anticipé avec coupon accumulé.
- **Barrière de protection ($60\% S_0$)** : À maturité, le capital est garanti tant que le sous-jacent ne franchit pas la barrière baissière.

---

## 6. Analyse Didactique de vos Résultats Numériques

Voici l'explication détaillée des résultats obtenus lors des benchmarks de votre repo sur l'action **NVIDIA ($NVDA$)** ($S_0 = 211.94\$$, $\sigma = 36.48\%$):

1. **Temps d'Inférence (PINN $0.61\text{ ms}$ vs Monte Carlo $554.4\text{ ms}$)** :
   * Le modèle PINN / Neural SDE est **900 fois plus rapide** pour évaluer un lot de prix en production qu'une simulation Monte Carlo classique.
2. **Précision du Modèle (MAE $0.2627\$$ sur l'option vanille)** :
   * Une erreur de 26 centimes sur une option de $20\$$, soit une précision relative de **96.6% à 99.4%**, parfaitement suffisante pour le pre-trade pricing et le screening massif de portefeuilles.
3. **Autocall Phoenix ($NVDA$)** :
   * Le prix théorique calculé ($0.99\$$ par dollar de nominal) indique un produit structuré très attractif avec une probabilité de rappel anticipé de **71.1%** et un risque de perte en capital limité à **7.6%**.
