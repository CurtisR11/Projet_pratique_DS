# Projet_pratique_DS

Projet Python – Guide de lecture
Bonjour,
 Afin de faciliter la lecture et la compréhension de notre projet, nous avons centralisé l’ensemble des éléments sur un dépôt GitHub.
Dans ce dépôt, vous trouverez un dossier nommé projet_python_main. Ce dossier contient tous les TP réalisés dans le cadre du Projet, ainsi que le modèle entraîné lors du TP7.

Exécution du projet
L'exécution complète peut se faire via le fichier main.py, qui orchestre l’ensemble des TP. En lançant ce fichier :
Le TP1 va générer automatiquement un dossier nommé projet_python, contenant :
Un dossier data_tp1 :
Un fichier ratio.csv
Un dossier companies, contenant un fichier .csv par entreprise étudiée
Le TP2, et les suivants, créeront à leur tour un dossier output au sein du folder projet_python, à l’intérieur duquel chaque TP générera son propre sous-dossier contenant les résultats et données liés à ce TP.

En cas de problème avec main.py
Si vous rencontrez des difficultés lors de l’exécution du main.py, nous avons prévu plusieurs alternatives :
Un notebook nommé Summary.ipynb, qui sert de récapitulatif final pour certaines compagnies et peut être lancé indépendamment pour accéder à certaines données clés extraites des TP. En principe, ce code s’exécute automatiquement avec le main.py (il se situe à la fin de celui-ci), mais il peut être utilisé séparément si nécessaire.
Nous avons également fourni un second dossier projet_python(exemple) déjà généré suite à l’exécution du projet sur notre environnement. Il vous permet de consulter directement les résultats sans avoir besoin de relancer les scripts.
Enfin, un notebook global TP_final.ipynb regroupe tous les TP : il est possible de les exécuter un par un manuellement, si jamais l’exécution automatique ne fonctionne pas.

⚠️ À lire / Important
Compatibilité Colab : Certains TP fonctionnent mieux sur Google Colab que sur VS Code, même si le projet a initialement été développé sous VS Code. C’est notamment le cas :
du TP3
et du TP7
TP7 : Ce TP n’est pas exécuté automatiquement dans le main.py, car il est trop long à lancer. À la place, un modèle préentraîné est fourni et utilisé directement dans le TP8.
Données partielles selon les TP : Dans certains TP (par exemple TP4, TP5 et TP8), toutes les compagnies n'ont pas été utilisées. Cela est volontaire, car certaines analyses sont plus performantes sur un sous-ensemble. Il est donc normal que tous les TP ne traitent pas toutes les compagnies.
Structure des dossiers : Pour que le projet fonctionne correctement, le dossier projet_python_main doit impérativement être placé sur le bureau (Desktop). C’est ainsi que les chemins ont été définis dans le code. Lors de l'exécution, un dossier projet_python sera automatiquement généré également sur le bureau, et contiendra tous les résultats.
Compatibilité macOS : Toute notre équipe travaillant sur macOS, nous avons utilisé la librairie os pour la gestion des chemins et l’enregistrement des fichiers, afin de garantir une compatibilité optimale dans cet environnement.

N'hésitez pas à nous contacter si certains fichiers ne sont pas lisibles ou si vous avez besoin d’éclaircissements sur le déroulement du projet.
Bien cordialement,
 Nassim Idamer Matthieu Rafatjah Curtis Roan
