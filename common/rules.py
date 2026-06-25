"""Rule extraction
  - extract_rules(net): Pour chaque neurone de sortie, extraire les règles qui l'activent afin d'aider à l'interprétabilité
  - sharpness regularization: Faire bouger b vers 0 ou 1, pour que le switch soit plus "net" (plus proche d'une fonction booléenne).
  - prune_rules(rules): Retirer les règles pas utilisées ou redondantes (comparer les R2)
"""
