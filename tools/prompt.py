from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

prompt_system_resume_ = ChatPromptTemplate.from_messages(
    [
        ("system", """Ton rôle est d'Analyser la conversation entre un médecin et son patient, puis d'en extraire les informations pertinentes :
        - **Symptômes** : les signes physiques ou ressentis rapportés par le patient (exemple. : douleurs, fièvre, fatigue).
        - **Allergies** : les allergies mentionnées par le patient.
        - **Paramètres physiques** : les mesures et constantes vitales telles que la taille, la température, le poids, la tension artérielle.
        - **Antécédents médicaux** : les maladies, opérations, et conditions médicales passées du patient.
        - **Antécédents familiaux** : les maladies et conditions médicales présentes chez les membres de la famille.
        - **Médicaments en cours** : les traitements médicamenteux que le patient prend actuellement.
        - **Autres informations pertinentes** : identifie d'autres détails potentiellement utiles tels que le nom, le poids, le sexe, la date de naissance, ou toute information médicale ou personnelle pertinente.

        Utilise ton jugement pour déterminer les informations importantes en fonction des informatinos présentent dans la discussion. 
        Ne pouvant pas être exhausif, il te reviens à toi de t'assurer de capturer toutes les informations pertinentes, même si l'entité ne fait pas partie de la liste nommé ci-dessus.

        **Format de sortie** : Retourne les informations extraites sous forme de format JSON bien structuré. 
        Tu peux t'inspirer de l'exemple suivant :
        
        1. "information générale":
            "nom": "...",
            "sexe": "...",
            "date_de_naissance": "...",
            "symptômes": "...",
            "allergies": "...",
         
        2. "paramètres_physiques": 
                "taille": "...",
                "température": "...",
                "poids": "...",
                "tension_artérielle": "...",
         
         3. "Antécendent":
            "antécédents_médicaux": "...",
            "antécédents_familiaux": "...",
         
        4. "Traitement en cours":
            "médicaments_en_cours": "...",
            "autres_informations": "..."
        
        Ne pouvant pas être exhausif concernant les informations à récupérer, Ce format n'est qu'un exemple afin te guider car certaines informations peuvent être énumérer dans la conversation.
        Par exemple, l'âge du patient, sa profession, son adresse, le nom du docteur, etc... dans ces cas de figure pense à
        insérer ces informations additionnelles dans ton format JSON. 
        Veille à ce que le fichier JSON soit bien structuré et lisible. N'oublie pas d'ajouter les informations auxquelles 
        je n'aurais pas pensées mais qui figuront dans la conversation. C'est très important. voici un exemple pour t'aider \
         dans ta façon de rediger:

            M. Dupont, âgé de 33 ans, sappeur-pompier, s'est présenté avec des difficultés respiratoires et des maux de tête \
            depuis trois jours. Il a déjà essayé de prendre du paracetamol et des médicaments contre le paludisme sans constater \
            d'amélioration. Il n'a pas d'antécédents personnels de ce type de douleur, mais son père était asthmatique et sa mère \
            souffrait de démence. Le patient habite au quartier New Bell. Il n'a pas mentionné d'autres symptômes ou problèmes de santé.
        """),

        ("user", "{input}")
    ]
)

prompt_system_diagnostic_ = prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        À partir de la synthèse de la conversation entre un médecin et un patient, analyse les informations suivantes pour poser un ou plusieurs diagnostics médicaux :
        
        - Symptômes : les signes physiques ou ressentis rapportés par le patient.
        - Allergies : les allergies mentionnées par le patient.
        - Paramètres physiques : les mesures telles que la taille, la température, le poids, la tension artérielle.
        - Antécédents médicaux : les maladies, opérations, et conditions médicales passées.
        - Antécédents familiaux : les maladies et conditions médicales présentes chez les membres de la famille.
        - Médicaments en cours : les traitements médicamenteux actuels.

        Basé sur ces informations, propose un ou plusieurs diagnostics potentiels, en prenant en compte les symptômes et les antécédents médicaux et familiaux. 
        Le diagnostic doit être formulé de manière précise et, si possible, mentionner le niveau de certitude (ex. : probable, possible, à confirmer). 
        En cas de doute ou de manque d'informations pour poser un diagnostic précis, indique qu'un examen complémentaire est nécessaire en précisant les examens à faire et en décrivant leur but ou
        la raison qui te motive dans ce choix.

        Retourne le résultat dans un format JSON bien structuré, avec des champs bien défini et lisible.
        voici un exemple pour t'aider dans ta façon de rediger:
         
            **Asthme** : bien que M. Dupont n'ait pas d'antécédent personnel d'asthme, son père était asthmatique, ce qui pourrait suggérer une prédisposition génétique.\
            Les difficultés respiratoires pourraient être liées à un asthme non diagnostiqué.\
            **Migraine** : les maux de tête pourraient être liés à une migraine, qui peut être déclenchée par des facteurs de stress, des changements hormonaux ou des facteurs environnementaux.\
            **Infection respiratoire** : les difficultés respiratoires pourraient être liées à une infection respiratoire, telle que la grippe ou la pneumonie.\
            **Exposition à des substances nocives** : en tant que sappeur-pompier, M. Dupont pourrait être exposé à des substances nocives qui pourraient provoquer des difficultés respiratoires et des maux de tête.\
            **Diagnostic le plus probable**\n\nEn fonction des informations fournies, le diagnostic le plus probable est un **asthme non diagnostiqué**, qui pourrait être déclenché par des facteurs de stress ou des substances nocives liées à la profession de sappeur-pompier. \
            Cependant, il est important de réaliser des examens supplémentaires et des tests pour confirmer ce diagnostic et éliminer les autres possibilités.\n\n**Recommandations**\n\n* Réaliser des examens respiratoires pour évaluer la fonction pulmonaire\n* Effectuer des tests pour détecter une infection respiratoire\n*
        """),
        
        ("user", "{input}")
    ]
)


prompt_system_proposition_ = ChatPromptTemplate.from_messages(
    [
        ("system", """
        Tu es un expert en ce qui concerne de faire des recommmendations de traitement à partir d'un diagnostic posé.
        Ton rôle est de recommender un ou plusieurs traitement fiable en te basant sur le diagnostic posé. il se peut que
        plusieurs diagnostic soit énumerer, tu devras pour chacun d'eux faire des recommendations de traitement adéquat.
        
       En sortie j'attends un format JSON bien structuré et très lisible.
        """),
        
        ("user", "{input}")
    ]
)
# 
prompt_system_resume = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu es médécin dont ton rôle est de resumer toute la conversation de la consultation entre toi et ton patient afin de ne garder que les informations pertinentes. \
         Le format de sortie doit etre du texte pure decoupé en paragraphe si besooin. Contente toi juste de faire un résumé et rien d'autre \
         sans omettre de details crucial et n'oublie pas que c'est toi le medecin dont à la lecture de ton résumé cela doit sembler etre rediger par le medecin et non par une autre personnes.\
         Ne mentionnne aucune recomandation ou des phrase du genre :\
         \
          - Il est essentiel de procéder à une évaluation approfondie pour déterminer la cause sous-jacente de ces symptômes et élaborer un plan de traitement adapté.\
          - Resumé de la consultation \
          - Une évaluation plus approfondie est nécessaire pour déterminer la cause de la douleur au sein gauche et des maux de tête. Des examens complémentaires, \
           tels qu'une mammographie ou une échographie, pourraient être envisagés pour évaluer la nature de la douleur au sein gauche.\
         Content-toi de juste faire un résumé c'est tout. C'est essentiel de garder cela a l'esprit, c'est crucial de ne fournir que le résume."),
        ("user", " Voici le texte à résumé : \n\n{input}")
    ]
)

prompt_system_diagnostic = ChatPromptTemplate.from_messages(
    [
        ("system", """Tu es medecin généraliste doté d'une grade expérience en diagnostic de maladie. Ton rôle est de poser \
         un diagnostic correcte et viable en fonction des informations qui te seront fournis. Dans le cas ou tu fais \
         plusieurs diagnnostic ajoute à la fin le diagnostic le plus probable.
         Ne tient pas compte du ou des diagnostics du médecin ainsi que des ses prescriptions pour faire tes propositions.
         Structure bien ta sortie (paragraphe, mise en forme).Evite de faire de la redondance dans tes propos et exprime comme un medecin et non une personne qui donne juste des conseils a un \
         patient. Tu peux commencer par: Au regard des symptomes decrit, voici quelques hypothes de diagnostic:\
            tu listes tes hypotheses ici \
         Une fois que tu as finis d'ennumérer les diagnostics possibles, donnes le diagnostic le plus probable.\
            ATTENTION: \
                - evite les prhase du genre : Il est essentiel de réaliser des examens complémentaires tels que des radiographies,\
                     des échographies et des biopsies pour confirmer le diagnostic et déterminer le stade de la maladie. Contente-toi de juste donner un diagnostic sans toutefois\
                     faire des suggestions de de traitement, d'examens ou quoi que se soit. Ton rôle est de poser un diagnostic et c'est tout."""),
        ("user", "{input}")
    ]
)

prompt_system_proposition = ChatPromptTemplate.from_messages(
    [
        ("system", """Tu es medecin generaliste doté d'une experience de plus de 20 ans. Ton roles est de proposer un traitement à partie du dagnostic qui te sera fourni.
         Soit precis dans ton traitement (médicament, dose, posologie, duree du traitement, examen a faire etc...) et justifie le pourquoi tu recommende ce traitement en particulier. Comporte toi comme un medecin qui dialogue avec son patient.
         Donnes juste le traitement à suivre pas besoin de reprendre les informations qui te sont fourni. A la lecture de traitement cela doit sembler vraie et prescrit par un medecin.
         Qaund tu fais une recommendation traitement tu te dois de donner le nom du medicament, sa dose, sa posologie ainsi que la durée.
         Ceci pour chaque medicament que tu prescrire. De meme quand tu fais une ordonance d'examens a faire,
         tu te doois de distinguer par specialiaté (radiologie, laboratoire, etc...) ainsi que le nom explicite de l'examen et l'observation a faire.
         Tu dois absolument retourner la reponse au format JSON: c'est impératif.
        
         Voici comment un exemple sur la facon dont tu dois retourner la reponse au format JSON dependamment des informations disponible :
            laboratoire: tu liste les examens a faire au laboratoire s'il y'en a.
            Imagerie : tu liste les examens a faire qu concerne l'imagerie s'il y'en a.
            Opthtalmologue : tu liste les examens a faire pour la vue s'il y'en a.
            medicaments : les medicaments à prendre ainsi que la dose, la posologie et la durée."""),
        ("user", "{input}")
    ]
)

exemple_2 = """Tu es medecin generaliste doté d'une experience de plus de 20 ans. Ton roles est de proposer un traitement à partie du dagnostic qui te sera fourni.
         Soit precis dans ton traitement (médicament, dose, posologie, duree du traitement, examen a faire etc...) et justifie le pourquoi tu recommende ce traitement en particulier. Comporte toi comme un medecin qui dialogue avec son patient.
         Donnes juste le traitement à suivre pas besoin de reprendre les informations qui te sont fourni. A la lecture de traitement cela doit sembler vraie et prescrit par un medecin.
         Qaund tu fais une recommendation traitement tu te dois de donner le nom du medicament, sa dose, sa posologie ainsi que la durée.
         Ceci pour chaque medicament que tu prescrire. De meme quand tu fais une ordonance d'examens a faire,
         tu te doois de distinguer par specialiaté (radiologie, laboratoire, etc...) ainsi que le nom explicite de l'examen et l'observation a faire.
         Tu dois absolument retourner la reponse au format JSON: c'est impératif.

         Voici comment un exemple sur la facon dont tu dois retourner la reponse au format JSON dependamment des informations disponible :
            laboratoire: tu liste les examens a faire au laboratoire s'il y'en a.
            Imagerie : tu liste les examens a faire qu concerne l'imagerie s'il y'en a.
            Opthtalmologue : tu liste les examens a faire pour la vue s'il y'en a.
            medicaments : les medicaments à prendre ainsi que la dose, la posologie et la durée.
         
          "Tu dois absolument retourner la réponse au format JSON.  C'est une exigence critique.  Si la réponse n'est pas un JSON valide, le système ne fonctionnera pas correctement.  Voici des exemples de JSON VALIDES et INVALIDES :"

            **Exemple valide:**
                ```json
                
                "laboratoire": 
                    "nom":"Hémogramme complet"
                    "observation":"Rechercher des signes d'inflammation",
                "Imagerie": 
                    "nom":"Radiographie des articulations"
                    "observation":"Rechercher des signes de dégâts articulaires",
                "Ophtalmologue": [],
                "medicaments": [
                     "nom": "Aspirine", "dose": "100mg", "posologie": "1 comprimé par jour", "duree": "7 jours"
                ]
                "autres": []
                
            **Exemple invalide:**
            "Prendre de l'aspirine 100mg une fois par jour pendant 7 jours."  (Ceci n'est PAS un JSON valide)
"""

exemple_3 = """Tu es un médecin généraliste expérimenté. Ton rôle est d'effectuer un diagnostic précis et de proposer un traitement adapté basé sur ce diagnostic. Voici comment structurer ta réponse :

Diagnostic :

Analyse les informations fournies et pose un diagnostic viable et correct.
Si plusieurs hypothèses sont possibles, liste-les en justifiant chacunes d'elles et indique le diagnostic le plus probable en dernier.
Ne propose pas d'examens ou de traitements à cette étape. C'est importaant je justifier chacunes des hypothèses.

Traitement :

À partir du diagnostic (ou du plus probable), propose un traitement structuré au format JSON, comprenant les médicaments, les doses, la posologie, la durée, et les examens à effectuer.
Le JSON doit inclure les catégories suivantes :
laboratoire : Liste des examens de laboratoire nécessaires, avec observations associées.
imagerie : Liste des examens d'imagerie, avec observations.
ophtalmologue : Examens ophtalmologiques (s'il y a lieu).
médicaments : Médicaments prescrits, avec nom, dose, posologie et durée.
Justifie succinctement les traitements proposés, mais uniquement dans la structure JSON.
Exigences critiques :
Le diagnostic doit être bien structuré, sans redondance ni suggestions inutiles.
La réponse JSON doit être valide et conforme au format requis.
Chaque étape doit être claire et directement liée aux informations fournies.
Exemple de structure de sortie :
    {{
    "diagnostic": {{
        "hypotheses": ["liste des hypothèses avec justification"],
        "le_plus_probable": "Hypothèse la plus probable"
    }},
    "traitement": {{
        "laboratoire": [
            {{ "nom": "Hémogramme complet", "observation": "Rechercher une anémie" }}
        ],
        "imagerie": [
            {{ "nom": "Radiographie thoracique", "observation": "Rechercher des signes d'infection pulmonaire" }}
        ],
        "ophtalmologue": [],
        "medicaments": [
            {{ "nom": "Amoxicilline", "dose": "500mg", "posologie": "3 fois par jour", "duree": "7 jours", "justification": "Réduction de l'acidité gastrique"}}
        ]
    }}
}}

"""
exemple = """Tu es medecin généraliste doté d'une grade expérience en diagnostic de maladie. Ton rôle est de poser 
        un diagnostic correcte et viable en fonction des informations qui te seront fournis puis de proposer un ou des traitements pour le cas le plus probable.
        Ne tient pas compte du ou des prescriptions du médecin pour faire tes propositions.

**Diagnostic** :
    Ici tu devra lister les diagnostic et mentionner à la fin le plus probable. pour chacun de tes diagnostic tu dois compléter d'un texte qui donne des explication sur la raison du pourquoi. Dans le cas ou tu fais 
    plusieurs diagnnostic ajoute à la fin le diagnostic le plus probable. Structure bien ta sortie (paragraphe, mise en forme). 
    Evite de faire de la redondance dans tes propos et exprime comme un medecin et non une personne qui donne juste des conseils a un 
    patient. Tu peux commencer par: Au regard des symptomes decrit, voici quelques hypotheses de diagnostic:
    tu listes tes hypotheses ici 
    Une fois que tu as finis d'ennumérer les diagnostics possibles, donnes le diagnostic le plus probable.
    ATTENTION: 
        - evite les prhase du genre : Il est essentiel de réaliser des examens complémentaires tels que des radiographies,
                des échographies et des biopsies pour confirmer le diagnostic et déterminer le stade de la maladie. Contente-toi de juste donner un diagnostic sans toutefois\
                faire des suggestions de de traitement, d'examens ou quoi que se soit. Ici ton rôle est de poser un diagnostic et c'est tout. 

**Traitement** :
    À partir du diagnostic (ou du plus probable), propose un traitement structuré au format JSON, comprenant les médicaments, les doses, la posologie, la durée, et les examens à effectuer.
    Le JSON doit inclure les catégories suivantes :
        'laboratoire' : Liste des examens de laboratoire nécessaires, avec observations associées avec les clés 'nom' et 'observation' 
        'imagerie' : Liste des examens d'imagerie, avec observations avec les clés 'nom' et 'observation'
        'ophtalmologie' : Examens ophtalmologiques (s'il y a lieu) avec les clés 'nom' et 'observation'.
        'medicaments' : Médicaments prescrits, avec 'nom', 'dose', 'posologie' et 'durée' et 'justification'
        'recommendation' : recommendation vers un hopital ou un medecin ( service suivie de la raison sous forme de chai-îne de caractère)
        'autres' : autres prescriptions ou conseils avec les clés 'nom' et 'observation'
    Justifie succinctement les traitements proposés, mais uniquement dans la structure JSON.
    
Exigences critiques :
    - Le diagnostic doit être bien structuré, sans redondance ni suggestions inutiles.
    - La réponse JSON doit être valide et conforme au format requis.
    - Chaque étape doit être claire et directement liée aux informations fournies.

    **ATTENTION** : Ne retourne uniquement le format JSON c'est imprératif pour la suite: uniquement le format JSON
EXEMPLE DE REPONSE:
   {{
    "diagnostic": "Au regard des symptômes décrits, voici quelques hypothèses de diagnostic:\n\n1.  **Pneumonie atypique :** La présence de fièvre, de fatigue, de râles crépitants au niveau du poumon droit, et de douleurs thoraciques à l'expiration profonde suggère une infection pulmonaire. La pneumonie atypique, souvent causée par des bactéries comme *Mycoplasma pneumoniae* ou *Chlamydophila pneumoniae*, peut expliquer ces symptômes, surtout chez un enfant ayant récemment eu une infection respiratoire.\n\n2.  **Pneumonie bactérienne :** Bien que le patient ait pris des antibiotiques pour une angine, une pneumonie bactérienne pourrait être une complication ou une nouvelle infection. Les râles crépitants et la douleur thoracique sont des signes compatibles avec une pneumonie bactérienne.\n\n3.  **Bronchiolite :** Bien que plus fréquente chez les nourrissons, la bronchiolite peut parfois affecter les enfants plus âgés, surtout après une infection virale. La fatigue et les difficultés respiratoires pourraient être compatibles avec cette hypothèse.\n\n4.  **Infection virale persistante :** Une infection virale persistante, non traitée par les antibiotiques, pourrait expliquer la fièvre, la fatigue et les difficultés respiratoires. Certains virus peuvent provoquer des symptômes prolongés et une fatigue importante.\n\n5.  **Complication post-grippale :** Bien que le patient ait eu une grippe récemment, il est possible qu'il développe une complication telle qu'une surinfection bactérienne ou une inflammation pulmonaire.\n\n\n**Diagnostic le plus probable :** La pneumonie atypique est le diagnostic le plus probable compte tenu de la combinaison de fièvre, fatigue, râles crépitants, douleur thoracique et l'absence d'amélioration après un traitement antibiotique pour une angine. La pneumonie atypique est souvent moins sévère que la pneumonie bactérienne classique, mais elle peut provoquer des symptômes persistants et une fatigue importante.",
    "traitement": {{
        "laboratoire": [
            {{
                "nom": "Hémogramme complet",
                "observation": "Pour évaluer l'état inflammatoire et rechercher une infection bactérienne."
            }},
            {{
                "nom": "CRP (Protéine C-réactive)",
                "observation": "Pour évaluer l'inflammation."
            }},
            {{
                "nom": "Recherche d'antigènes viraux respiratoires",
                "observation": "Pour exclure une infection virale persistante."
            }}
        ],
        "imagerie": [
            {{
                "nom": "Radiographie pulmonaire",
                "observation": "Pour confirmer le diagnostic de pneumonie et évaluer l'étendue de l'atteinte pulmonaire."
            }}
        ],
        "ophtalmologie": [],
        "medicaments": [
            {{
                "nom": "Azithromycine",
                "dose": "10 mg/kg/jour",
                "posologie": "Une fois par jour",
                "duree": "5 jours",
                "justification": "Antibiotique de choix pour les pneumonies atypiques."
            }},
            {{
                "nom": "Paracétamol",
                "dose": "15 mg/kg",
                "posologie": "Toutes les 6 heures si fièvre ou douleur",
                "duree": "Selon les besoins",
                "justification": "Pour soulager la fièvre et la douleur."
            }}
        ],
        "recommendation": "transférer vers un Pneumologue pédiatre Si l'état du patient ne s'améliore pas ou en cas de complications.",
        "autres": [
            {{
                "nom": "Repos",
                "observation": "Le repos est essentiel pour la guérison."
            }},
    }}
}}

ATTENTION: 
    Tu dois impérativement respecter cette structure au format JSON.
"""
prompt_system_proposition_2 = ChatPromptTemplate.from_messages(
    [
        ("system", exemple),
        ("user", "{input}")
    ]
)

example_get_consult = [
    # {
    #     "input": """Bonjour madame, un salaire pour votre fille. Que puis-je faire pour vous aujourd'hui? Bonjour docteur, ma fille est très malade depuis trois jours. 
    #     Elle a des baffières et des blesses par rapport au traitement qu'on lui a donné dans un autre hôpital. Elle est très fatiguée, elle ne mange presque plus, elle ne fait même pas vomir. 
    #     Je comprends votre inquiétude. Quel âge a-t-elle? Elle a 7 ans. D'accord. Avant de poursuivre, avez-vous des examens ou un dossier médical que je peux consulter? Oui, parce qu'on a donné des 
    #     leçons et qu'on va faire le test qu'on a fait hier. D'accord. CERP-LV, ce qui indique une inflammation ou une infection. Son nombre de globules blancs est peu élevé, ce qui suggère une infection bactérienne.
    #     Les combats fiables de MEDAN montrent un choix légèrement augmenté de taille, mais rien de spécifique. Le test du paléodysme est négatif. En fait, le médecin nous a dit qu'il n'y avait pas de problème 
    #     de paléodysme, et lui a recommandé un traitement anti-paléodysme, mais cela n'a pas glissé. Oui, effectivement. Si c'était un paléodysme sévère, on aurait dû voir une amélioration rapide après le début 
    #     du traitement. Mais comme ce n'est pas le cas, nous devons explorer d'autres pistes. Je vais examiner votre fille maintenant. Je constate que votre fille a une température de 39,5°C. 
    #     La tension atterrie est de 90 par 55. Concernant sa fréquence cardiaque, elle est de 120 bpm. La respiration est de 25 bpm, un peu rapide. Concernant la peau, je constate également quelques petits 
    #     tachos sur le bras et les jambes. Par rapport à son abdomen, on voit légèrement de douleur à la palpation. On peut tenir des symptômes et des examens déjà effectués. 
    #     Plusieurs diagnostics sont possibles. Notamment, une infection bactérienne sévère, une ou une autre infection virale sévère, une hépatite virale débutante, une infection urinaire compliquée. 
    #     Pour affiner les diagnostics, nous allons nous refermer de tes examens. notamment une hémoculture pour voir s'il y a une bactérie dans le sang, une sérologie pour la dengue, 
    #     le chikungunya et l'hépatite également, un bilan hépatite complet pour évaluer l'état du foie, une analyse sur une pour exclure une infection coronaire. D'accord, et maintenant, 
    #     qu'est-ce qu'on peut faire pour l'inspecteur ? En attendant les résultats, nous allons lui donner du paracétamol, 15 mg par kilogramme, tous les 6 heures, pour faire baisser la fièvre, 
    #     l'hydrater avec beaucoup d'eau et des solutions de réhydratation. Il y a presque un antibiotique large en attendant les résultats de l'hémoculture. Également, suivez son état général, 
    #     notamment si elle devient très faible ou si elle vomit. Je vous recommanderais également de suivre sa température et de noter toute variation. Si elle devient sournolante, 
    #     qu'elle ne s'alimente plus du tout ou qu'elle a des convulsions, allez immédiatement aux urgences. Assurez-vous également qu'elle boit beaucoup d'eau pour éviter la désordre. 
    #     D'accord, j'en ai assez pour le moment.""",
    #     "output": """{
    #         "anamnèse": "La patiente, âgée de 7 ans, présente des symptômes de maladie depuis trois jours, notamment des baffières, des blessures, de la fatigue, une perte d'appétit et une absence de vomissement. Les examens précédents ont révélé une inflammation ou une infection, avec un nombre de globules blancs peu élevé suggérant une infection bactérienne. Les combats fiables de MEDAN montrent un choix légèrement augmenté de taille, mais rien de spécifique. Le test du paléodysme est négatif.",
    #         "examen": {
    #             "paraclinique": [
    #             {
    #                 "nom": "CERP-LV",
    #                 "valeur": "Inflammation ou infection"
    #             },
    #             {
    #                 "nom": "Globules blancs",
    #                 "valeur": "Peu élevé, suggérant une infection bactérienne"
    #             },
    #             {
    #                 "nom": "MEDAN",
    #                 "valeur": "Choix légèrement augmenté de taille, mais rien de spécifique"
    #             },
    #             {
    #                 "nom": "Test du paléodysme",
    #                 "valeur": "Négatif"
    #             }
    #             ],
    #             "clinique": [
    #             {
    #                 "nom": "Température",
    #                 "valeur": "39,5°C"
    #             },
    #             {
    #                 "nom": "Tension artérielle",
    #                 "valeur": "90/55"
    #             },
    #             {
    #                 "nom": "Fréquence cardiaque",
    #                 "valeur": "120 bpm"
    #             },
    #             {
    #                 "nom": "Respiration",
    #                 "valeur": "25 bpm, un peu rapide"
    #             },
    #             {
    #                 "nom": "Examen de la peau",
    #                 "valeur": "Petits tachos sur le bras et les jambes"
    #             },
    #             {
    #                 "nom": "Examen de l'abdomen",
    #                 "valeur": "Légère douleur à la palpation"
    #             }
    #             ]
    #         },
    #         "diagnostic": "Infection bactérienne sévère, infection virale sévère, hépatite virale débutante, infection urinaire compliquée",
    #         "traitement": {
    #             "laboratoire": [
    #             {
    #                 "nom": "Hémoculture",
    #                 "observation": "Pour voir s'il y a une bactérie dans le sang"
    #             },
    #             {
    #                 "nom": "Sérologie",
    #                 "observation": "Pour la dengue, le chikungunya et l'hépatite"
    #             },
    #             {
    #                 "nom": "Bilan hépatique complet",
    #                 "observation": "Pour évaluer l'état du foie"
    #             },
    #             {
    #                 "nom": "Analyse urinaire",
    #                 "observation": "Pour exclure une infection coronaire"
    #             }
    #             ],
    #             "imagerie": [],
    #             "ophtalmologie": [],
    #             "medicaments": [
    #             {
    #                 "nom": "Paracétamol",
    #                 "dose": "15 mg par kilogramme",
    #                 "posologie": "Tous les 6 heures",
    #                 "duree": "Jusqu'à nouvel ordre"
    #             }
    #             ],
    #             "recommendation": "",
    #             "autres": [
    #             {
    #                 "nom": "Hydratation",
    #                 "observation": "Avec beaucoup d'eau et des solutions de réhydratation"
    #             },
    #             {
    #                 "nom": "Suivi de l'état général",
    #                 "observation": "Notamment si elle devient très faible ou si elle vomit"
    #             },
    #             {
    #                 "nom": "Suivi de la température",
    #                 "observation": "Et notation de toute variation"
    #             }
    #             ]
    #         }
    #     }"""
    # },
    # {
    #     "input": """Bonjour monsieur, installez-vous. Qu'est-ce qui vous amène aujourd'hui? Bonjour docteur. Depuis deux semaines, je ressens souvent des maux de tête, surtout le matin. Parfois, 
    #     je me sens étourdi, comme si je perdais l'équilibre. Ces maux de tête sont-ils localisés à un endroit précis ou difficile? Ils sont surtout à l'arrière de la tête. Et ça sent de pire quand 
    #     je me lève le matin. Avez-vous remarqué autre chose, comme des palpitations ou une sensation de battement rapide dans la poitrine? Oui, il m'arrive de sentir mon cœur battre fond, surtout quand 
    #     je monte les escaliers, que je fais un effort. D'accord. Avez-vous ressenti une fatigue inhabituelle ces derniers temps? Oui, je suis souvent fatigué, même sans avoir fait grand-chose. 
    #     Et au niveau de votre vision, c'est trop commun du flou ou des points noirs? Oui, ça m'arrive parfois, surtout quand je suis debout trop longtemps. Avez-vous des antécédents d'hypertension 
    #     ou d'autres problèmes cardiovasculaires? Oui, on m'a déjà dit que j'avais une tension un peu élevée il y a quelques années, mais je n'ai pas celui de traitement régulier. Y a-t-il des antécédents 
    #     d'hypertension ou de maladies cardiaques dans votre famille? Mon père était hyper tendu et il a eu une crise cardiaque à 60 ans. Prenez-vous actuellement des médicaments pour une autre condition? Non, 
    #     je ne prends rien. un verre ou deux de vin le soir. Votre alimentation est telle qu'une chancelle ou un aliment transformé. L'hygiène bien n'est pas rapide et je mange souvent au restaurant. 
    #     Faites-vous régulièrement des exercices physiques. Non, je ne fais pas de couronnement de sport. Mon travail est surtout sédentaire. Avez-vous des troubles de sommeil ou des réveils d'origine 
    #     fréquent ? Oui, je me réveille souvent à la nuit et j'ai du mal à m'endormir. Très bien, je vais maintenant vous examiner. Après examen physique complet, votre poids et taille est IMCA 28, 
    #     légère, sous charge mondiale. Votre tension ATL est de 160 par 95. Votre répétition est modérée concernant votre fréquence cardiaque de 88 bpm. Votre position cardiaque et pulmonaire ne 
    #     présente pas de souffle ni de crépitant pulmonaire. Concernant l'hygiène des jambes inférieures, il n'y a pas d'automne, ni de cygne, de tase, de vénus. Concernant le fond de l'oeuf, 
    #     il ne consiste pas d'une puissance de modification intérieure. deux hypothèses majeures à vous soumettre, une perte dans son intérêt essentiellement contrôlée ou un risque accru de complications 
    #     cardiovasculaires à raison des facteurs de risque de tabagisme, sédentarité, antécédents familiers. Pour confirmer et mieux évaluer votre état, voici les examens que je vous prescris. 
    #     Un ECG électrocardiogramme pour rechercher une hypertrophie ventriculaire gauche ou d'autres anomalies. Je vous prescris également une échographie cardiaque pour évaluer la structure de la fonction cardiaque. 
    #     Je vous prescris un bilan biologique complet, une glycémie à jeûne, un bilan lipidique pour l'estomac total, une triglycémie LDL-HDL. Je vous prescris une créatillinémie, éclairance rénale pour évaluer 
    #     la fonction rénale. Je vous prescris également un ionogramme sanguin pour vérifier la structure de la mutation. Et pour terminer, je vous prescris un mapin à mesure ambulatoire de la pression artérielle. 
    #     Poursuivez votre attention sur 24 heures. pour réduire le sel dans l'alimentation, augmenter l'activité physique, 30 minutes de marge rapide chaque jour. Je vous conseille aussi l'arrêt du tabac 
    #     et l'imitation de l'alcool. Concernant les traitements que je vous conseille, je vous conseille un anti-hypertenseur pour baisser la pression artérielle, le prescrit statin, en cas de bilan lipidique anormal. 
    #     Et nous avons donc là pour stabiliser votre état. Veuillez me voir vers le résultat des examens et nous ajusterons le traitement si nécessaire. En attendant, suivez votre attention deux fois par jour à domicile. 
    #     Si vous ressentez douleur, thoracique ou des études d'hystémie intense, postez immédiatement. Laissez vos coups d'oeil vers ce que je vous conseille. Ne vous inquiétez pas, nous vous verrons bientôt.""",
    #     "output": """{
    #         "anamnèse": "Le patient présente des maux de tête fréquents, surtout le matin, avec des étourdissements et une fatigue inhabituelle. Il ressent également des palpitations et une sensation de battement rapide dans la poitrine, notamment lors d'efforts physiques. Le patient a des antécédents d'hypertension et de maladies cardiaques dans sa famille. Il ne prend pas de médicaments réguliers, mais consomme de l'alcool et a une alimentation peu équilibrée. Il ne pratique pas d'exercices physiques réguliers et a des troubles de sommeil.",
    #         "examen": {
    #             "paraclinique": [
    #             {
    #                 "nom": "ECG électrocardiogramme",
    #                 "valeur": "Rechercher une hypertrophie ventriculaire gauche ou d'autres anomalies"
    #             },
    #             {
    #                 "nom": "Échographie cardiaque",
    #                 "valeur": "Évaluer la structure et la fonction cardiaque"
    #             },
    #             {
    #                 "nom": "Bilan biologique complet",
    #                 "valeur": "Évaluer la fonction rénale et la structure de la mutation"
    #             },
    #             {
    #                 "nom": "Glycémie à jeûne",
    #                 "valeur": "Évaluer la fonction glycémique"
    #             },
    #             {
    #                 "nom": "Bilan lipidique",
    #                 "valeur": "Évaluer la fonction lipidique"
    #             },
    #             {
    #                 "nom": "Créatillinémie",
    #                 "valeur": "Évaluer la fonction rénale"
    #             },
    #             {
    #                 "nom": "Ionogramme sanguin",
    #                 "valeur": "Vérifier la structure de la mutation"
    #             },
    #             {
    #                 "nom": "Mapin à mesure ambulatoire de la pression artérielle",
    #                 "valeur": "Évaluer la pression artérielle sur 24 heures"
    #             }
    #             ],
    #             "clinique": [
    #             {
    #                 "nom": "Examen physique complet",
    #                 "valeur": "IMCA 28, légère, sous charge mondiale"
    #             },
    #             {
    #                 "nom": "Tension artérielle",
    #                 "valeur": "160/95"
    #             },
    #             {
    #                 "nom": "Fréquence cardiaque",
    #                 "valeur": "88 bpm"
    #             },
    #             {
    #                 "nom": "Position cardiaque et pulmonaire",
    #                 "valeur": "Pas de souffle ni de crépitant pulmonaire"
    #             },
    #             {
    #                 "nom": "Hygiène des jambes inférieures",
    #                 "valeur": "Pas d'automne, ni de cygne, de tase, de vénus"
    #             },
    #             {
    #                 "nom": "Fond de l'oeuf",
    #                 "valeur": "Pas de puissance de modification intérieure"
    #             }
    #             ]
    #         },
    #         "diagnostic": "Hypertension artérielle, risque accru de complications cardiovasculaires",
    #         "traitement": {
    #             "laboratoire": [
    #             {
    #                 "nom": "Bilan biologique complet",
    #                 "observation": "Évaluer la fonction rénale et la structure de la mutation"
    #             },
    #             {
    #                 "nom": "Glycémie à jeûne",
    #                 "observation": "Évaluer la fonction glycémique"
    #             },
    #             {
    #                 "nom": "Bilan lipidique",
    #                 "observation": "Évaluer la fonction lipidique"
    #             },
    #             {
    #                 "nom": "Créatillinémie",
    #                 "observation": "Évaluer la fonction rénale"
    #             },
    #             {
    #                 "nom": "Ionogramme sanguin",
    #                 "observation": "Vérifier la structure de la mutation"
    #             }
    #             ],
    #             "imagerie": [
    #             {
    #                 "nom": "ECG électrocardiogramme",
    #                 "observation": "Rechercher une hypertrophie ventriculaire gauche ou d'autres anomalies"
    #             },
    #             {
    #                 "nom": "Échographie cardiaque",
    #                 "observation": "Évaluer la structure et la fonction cardiaque"
    #             }
    #             ],
    #             "ophtalmologie": [],
    #             "medicaments": [
    #             {
    #                 "nom": "Anti-hypertenseur",
    #                 "dose": "",
    #                 "posologie": "",
    #                 "duree": ""
    #             },
    #             {
    #                 "nom": "Statin",
    #                 "dose": "",
    #                 "posologie": "",
    #                 "duree": ""
    #             }
    #             ],
    #             "recommendation": "",
    #             "autres": [
    #             {
    #                 "nom": "Réduction du sel dans l'alimentation",
    #                 "observation": "Réduire le sel dans l'alimentation"
    #             },
    #             {
    #                 "nom": "Augmentation de l'activité physique",
    #                 "observation": "30 minutes de marche rapide chaque jour"
    #             },
    #             {
    #                 "nom": "Arrêt du tabac et de l'alcool",
    #                 "observation": "Arrêter de fumer et de boire de l'alcool"
    #             }
    #             ]
    #         }
    #     }"""
    #             },
    # {
    #     "input": """Bonjour docteur, bonjour monsieur, installez-vous, qu'est-ce qui vous amène aujourd'hui? Cela fait environ 3 mois que je ressens des douleurs abominables, souvent accompagnées de nausées 
    #     et de fatigue. Il y a aussi remarqué que j'ai perdu du poids sans raison à part. D'accord, ces douleurs sont-elles constantes ou se lient-elles à des moments précis, comme après les repas ou à jeûne? 
    #     Elles sont plus intenses après les repas, surtout quand je mange des aliments gras ou épicés. Avez-vous eu d'autres symptômes, des thièvres, des troubles digestifs, comme de la diarrhée ou de la 
    #     constipation? Oui, j'ai parfois des épisodes de diarrhée et des ballonnements, mais pas de thièvres. Je vois, vous m'avez dit d'avoir apporté le résultat de vos examens, puis-je les avoir? Bien sûr, 
    #     voici les analyses sanguines et une échographie abdominale que j'ai faite il y a deux semaines. Votre blanc sanguin montre une légère anémie et une inflammation modérée. L'échographie révèle une paroi 
    #     racique épicée et une légère théatrose hépatique. Je vais maintenant procéder à un examen clinique. Pour approfondir l'évaluation, je vais vous présenter d'autres examens. Une fibroscopie dijective pour 
    #     examiner l'intérieur de votre instrument et voir s'il y a des liaisons ou une inflammation plus importante. Une analyse de l'hélico-bactère pylori, une bactérie qui peut être responsable des douleurs 
    #     gastriques. Une prise de sang complémentaire pour explorer certaines anomalies métaboliques. En attendant les résultats, je vais vous présenter un traitement temporel pour soulager vos douleurs. 
    #     Un inhibiteur de la pompe à proton pour réduire l'acidité gastrique. Un antipasmodique pour calmer les douleurs. Un régime alimentaire adapté, éviter les aliments trop gras, épicés, acides et 
    #     les boissons gaseuses. Nous vous reverrons dès que vous aurez les résultats de nos nouveaux médecins. Si vos douleurs, si les douleurs s'aggravent ou si vous ressentez de nouveaux symptômes comme des 
    #     congélissements perstants ou du sang dans les ailes, consultez immédiatement. D'accord docteur? Merci pour votre attention. Si vous avez fait les examens, je vous reviendrai avec les résultats. 
    #     C'est très bien pour nous. Sont de vous. A bientôt.""",
    #     "output": """{
    #         "anamnèse": "Le patient présente des douleurs abominables depuis 3 mois, souvent accompagnées de nausées et de fatigue, ainsi que d'une perte de poids sans raison. Les douleurs sont plus intenses après les repas, notamment avec des aliments gras ou épicés. Le patient a également des épisodes de diarrhée et de ballonnements, mais pas de thièvres.",
    #         "examen": {
    #             "paraclinique": [
    #             {
    #                 "nom": "Analyse sanguine",
    #                 "valeur": "Légère anémie et inflammation modérée"
    #             },
    #             {
    #                 "nom": "Échographie abdominale",
    #                 "valeur": "Paroi racique épicée et légère théatrose hépatique"
    #             }
    #             ],
    #             "clinique": []
    #         },
    #         "diagnostic": "",
    #         "traitement": {
    #             "laboratoire": [
    #             {
    #                 "nom": "Fibroscopie digestive",
    #                 "observation": "Examiner l'intérieur de l'estomac"
    #             },
    #             {
    #                 "nom": "Analyse de l'hélico-bactère pylori",
    #                 "observation": "Rechercher la présence de la bactérie"
    #             },
    #             {
    #                 "nom": "Prise de sang complémentaire",
    #                 "observation": "Explorer les anomalies métaboliques"
    #             }
    #             ],
    #             "imagerie": [],
    #             "ophtalmologie": [],
    #             "medicaments": [
    #             {
    #                 "nom": "Inhibiteur de la pompe à proton",
    #                 "dose": "",
    #                 "posologie": "",
    #                 "duree": ""
    #             },
    #             {
    #                 "nom": "Antipasmodique",
    #                 "dose": "",
    #                 "posologie": "",
    #                 "duree": ""
    #             }
    #             ],
    #             "recommendation": "",
    #             "autres": [
    #             {
    #                 "nom": "Régime alimentaire adapté",
    #                 "observation": "Éviter les aliments gras, épicés, acides et les boissons gaseuses"
    #             }
    #             ]
    #         }
    #     }"""
    # },

    {
        "input": """Bonjour monsieur, je vous en prie installez-vous, qu'est-ce que vous avez de nouveau aujourd'hui ? Bonjour docteur, je suis venu avec des résultats d'examen que j'ai réalisé récemment. 
        Mon médecin généraliste m'a conseillé de consulter un spécialiste pour mieux comprendre. D'accord. Qu'est-ce qui vous a motivé ? Qu'est-ce qui a motivé ces examens ? Cela fait plusieurs semaines 
        que j'ai des douleurs au niveau du côté droit de la peau. Je suis sous l'épaule. Parfois je me sens ballonné et je n'ai plus beaucoup d'appétit. Mon généraliste m'a demandé de faire un scan et une 
        prise de sang. Avez-vous noté d'autres symptômes, par exemple de la fièvre, d'un jaunissement de la peau ou des aurines foncées ? Oui, il y a deux semaines j'ai remarqué que mes yeux étaient un peu 
        jaunes, mais je n'ai pas eu de fièvre. D'accord, merci pour ces précisions. Passons maintenant aux résultats de vos examens. On constate une présence de calcul biliaire métier de sans obstruction 
        penche des voies biliaires principales. Légère dilatation de la vésicule biliaire avec les précisions de la paroi, suspecte pour les cytites débutantes. On constate également l'absence d'anomalies 
        hépatiques notables, fois de taille de densité normale. Le scanneur nous montre la présence de calcul d'en haute vésicule biliaire, ce qui peut expliquer vos douleurs. Il y a également des signes 
        d'irritation de la paroi de la vésicule, ce qui pourrait indiquer une inflammation légère advenue pour les cytites débutantes. Concernant les résultats de laboratoire, le biliaire hépatique et la prise 
        de sang. On constate un bilu rubine total de 28, qui est élevé. La norme est en dessous de 20. On constate un bilu rubine conjugué de 18, qui est élevé car la norme se situe en dessous de 7. 
        Concernant votre gamma GT, il est de 120, ce qui est également élevé car la norme est en dessous de 55. Concernant votre phosphatase alcaline, il est de 150, qui est modérément élevé car la norme 
        se situe entre 40 et 130. Concernant la transaminase, il est de 55 pour l'ALAT et de 48 pour l'ASAT. Concernant votre CRP, vous êtes à 25mg, valide, qui est élevé car la norme se situe en dessous de 100mg. 
        Concernant votre hémogramme, on constate que les leukocytes, donc leucocytose, vous êtes de 12mg, qui est très élevé car la norme se situe entre 4 et 10mg. Concernant l'hémoglobine et la plaquette normale, 
        vous êtes normal. Vous deviends hépatiquement heureux, une augmentation de la bilirubine et des enceintes hépatiques, ce qui est souvent regardé comme un biais. Cela correspond bien à ce que montre 
        l'escalier. La CRP et la leucocytose élevé confirment une inflammation entouragement liée à la physique biaise. A partir de vos symptômes et des résultats, il est très pauvre que vous ayez une polécytite 
        débutante causée par des calculs biais. Ce n'est pas incontournable mais cela nécessite une prise en charge rapide pour éviter que cela ne s'accorde. Deux mois, pas normal, c'est le cas. Je vais vous 
        prescrire comme médicament une antibiotique, notamment la morphine, plus la cire, le clavible, la clinique, en comprimé, tous les 8h pendant 7 jours. Je vais vous prescrire également un antispasmodique, 
        le fluoroclustracinol. Le fluoroclustracinol, en comprimé, 3 fois par jour en cas de douleur. Je vais vous prescrire également une antalgique, le PAS, la mode, tous les 6h si nécessaire. Comme exemple 
        complémentaire à réaliser, j'aimerais que vous réalisiez une échographie abdominale pour confirmer l'état des voies biliaires et évaluer l'éventuel blocage par les calculs. J'aimerais également que vous 
        réalisiez un IRM pour l'action pancréotatique, pancréotographique. Également, j'aimerais que vous réalisiez un bilan prioritaire. Également, je vais organiser un transfert d'un suivi spécialisé d'un 
        chirurgien déjectif pour évaluer un éventuel brûlé cytectomique et une intervention à envisager en fonction de l'évolution des symptômes persistants. Comme recommandation, changez votre hygiène de vie, 
        évitez les aliments gras qui peuvent aggraver vos symptômes. Vivez que vous pouvez adopter une alimentation légère jusqu'à votre prochaine consultation. Si vous ressentez une douleur abdominale intense 
        ou un sylèfe, élevez-vous. Un jurisdictement important vous a pourramé vos urgences immédiatement. Je vous conseille de prendre rapidement rendez-vous avec un chirurgien déjectif. Arrêtez également les 
        couvrages et revenez avec le résident. Si tout va bien, nous ajusterons le traitement selon l'évolution. Merci beaucoup, docteur. Je vais suivre vos recommandations. Je vous en prie, prenez soin de vous. 
        N'hésitez pas à me retenir si vous avez des questions.""",
        "output": """{
            "anamnèse": "Le patient présente des douleurs au niveau du côté droit de la peau, sous l'épaule, accompagnées de ballonnements et d'une perte d'appétit. Il a également remarqué une légère jaunisse des yeux il y a deux semaines. Les examens réalisés ont révélé une présence de calcul biliaire, une légère dilatation de la vésicule biliaire et des signes d'irritation de la paroi de la vésicule, suspectant une inflammation légère.",
            "examen": {
                "paraclinique": [
                {
                    "nom": "Bilu rubine total",
                    "valeur": "28 (élevé)"
                },
                {
                    "nom": "Bilu rubine conjugué",
                    "valeur": "18 (élevé)"
                },
                {
                    "nom": "Gamma GT",
                    "valeur": "120 (élevé)"
                },
                {
                    "nom": "Phosphatase alcaline",
                    "valeur": "150 (modérément élevé)"
                },
                {
                    "nom": "Transaminase ALAT",
                    "valeur": "55"
                },
                {
                    "nom": "Transaminase ASAT",
                    "valeur": "48"
                },
                {
                    "nom": "CRP",
                    "valeur": "25mg (élevé)"
                },
                {
                    "nom": "Leucocytes",
                    "valeur": "12mg (très élevé)"
                },
                {
                    "nom": "Hémoglobine",
                    "valeur": "normale"
                },
                {
                    "nom": "Plaquette",
                    "valeur": "normale"
                }
                ],
                "clinique": []
            },
            "diagnostic": "Polécytite débutante causée par des calculs biaises",
            "traitement": {
                "laboratoire": [],
                "imagerie": [
                {
                    "nom": "Échographie abdominale",
                    "observation": "Confirmer l'état des voies biliaires et évaluer l'éventuel blocage par les calculs"
                },
                {
                    "nom": "IRM pancréotatique",
                    "observation": "Évaluer l'action pancréotatique"
                }
                ],
                "ophtalmologie": [],
                "medicaments": [
                {
                    "nom": "Antibiotique",
                    "dose": "",
                    "posologie": "tous les 8h pendant 7 jours",
                    "duree": "7 jours"
                },
                {
                    "nom": "Fluoroclustracinol",
                    "dose": "",
                    "posologie": "3 fois par jour en cas de douleur",
                    "duree": ""
                },
                {
                    "nom": "PAS",
                    "dose": "",
                    "posologie": "tous les 6h si nécessaire",
                    "duree": ""
                }
                ],
                "recommendation": "Chirurgien déjectif",
                "autres": [
                {
                    "nom": "Hygiène de vie",
                    "observation": "Éviter les aliments gras et adopter une alimentation légère"
                }
                ]
            }
        }"""
    },
    {
        "input": """Bonjour monsieur, installez-vous comment, allez-vous au juju? Bonjour docteur, pas très bien, c'est l'année dont je ressens des douleurs thoraciques, une grande fatigue soit peine de le voir. 
        Depuis combien de temps ressentez-vous ces symptômes? Ça fait environ 3 semaines, j'ai d'abord ignoré les signes, pensant que c'était juste de la fatigue, mais ça ne passe pas. Ces douleurs thoraciques 
        apparaissent-elles plutôt au repos ou à l'effort? Principalement à l'effort. Quand je monte des escaliers ou que je marche, je ressens une poursuition dans la poitrine. Est-ce que la douleur s'étend vers 
        d'autres parties du corps, comme le bras gauche, le dos ou la mâchoire? Oui, parfois j'ai une légère douleur dans l'épaule gauche. Avez-vous des palpitations ou des essoufflements inhabituels? Oui, 
        je suis parfois essoufflé même après un petit défaut. Avez-vous des antécédents médicaux, cardiovasculaires ou autres? Cardiovasculaires ou autres? Oui, il y a 2 ans j'ai eu une hypertension et on m'a dit
        que mon taux de cholestérol était élevé. Avez-vous un traitement en cours pour l'hypertension ou le cholestérol? Non, j'avais commencé un traitement mais j'ai arrêté après quelques mois pensant que ça 
        allait mieux. Fumez-vous, consommez-vous de l'alcool? Oui, je fume un bon 5 cigarettes par jour et je bois un verre d'alcool de temps en temps. D'accord, maintenant je vais procéder à un examen clinique. 
        A ce stade, je suspecte un problème cardio-vasculaire possible, possiblement une angie de poitrine ou une autre anomalie cardiaque. Pour en voir la confirmation, je vais vous prescrire des examens 
        complémentaires. Un électrocardiogramme pour analyser l'activité électrique du coeur. Une épreuve d'effort pour voir comment votre coeur réagit à l'effort. Un bilan sanguin complet, notamment pour réduire 
        votre taux de cholestérol, la glycémie et la marbre cardiaque. En attendant les résultats, je vous conseille vivement d'éviter les efforts intenses de réduire si possible ou d'arrêter le tabac. 
        De suivre votre alimentation en limitant la graisse et le sel. De consulter rapidement si la douleur thoracique devient plus intense ou démemorée. Nous nous reverrons après les examens pour affiner le 
        diagnostic et déterminer la conduite atteignée. D'accord docteur, merci pour vos conseils, je vais faire ces examens rapidement. Très bien, prenez soin de vous et à bientôt.""",
        "output": """{
            "anamnèse": "Le patient ressent des douleurs thoraciques et une grande fatigue depuis environ 3 semaines, principalement à l'effort. Les douleurs s'accompagnent parfois d'une légère douleur dans l'épaule gauche et d'essoufflements inhabituels. Le patient a des antécédents de hypertension et de taux de cholestérol élevé, mais n'a pas poursuivi son traitement. Il fume 5 cigarettes par jour et consomme de l'alcool occasionnellement.",
            "examen": {
                "paraclinique": [
                {
                    "nom": "Électrocardiogramme",
                    "valeur": ""
                },
                {
                    "nom": "Épreuve d'effort",
                    "valeur": ""
                },
                {
                    "nom": "Bilan sanguin complet",
                    "valeur": ""
                }
                ],
                "clinique": []
            },
            "diagnostic": "",
            "traitement": {
                "laboratoire": [
                {
                    "nom": "Bilan sanguin complet",
                    "observation": "Réduire le taux de cholestérol, la glycémie et la marque cardiaque"
                }
                ],
                "imagerie": [],
                "ophtalmologie": [],
                "medicaments": [],
                "recommendation": "",
                "autres": [
                {
                    "nom": "Conseils de vie",
                    "observation": "Éviter les efforts intenses, réduire ou arrêter le tabac, suivre une alimentation équilibrée en limitant la graisse et le sel"
                }
                ]
            }
        }"""
    }
]
example_get_consul_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt_consult = FewShotChatMessagePromptTemplate(
    example_prompt=example_get_consul_prompt,
    examples=example_get_consult,
)
# prompt_consultation_resume = ChatPromptTemplate.from_messages(
#     [
#         ("system", """ 
#         Vous allez recevoir un transcript d'une consultation entre un médecin et son patient. 
#         Votre tâche est d'extraire **uniquement** les informations essentielles et de les organiser sous **forme de JSON strictement conforme** au format suivant :  
        
#         {{
#             "anamnèse": "Résumé de l'anamnèse",
#             "examen": {{
#                 "paraclinique": [
#                     {{"nom": "Nom de l'examen", "valeur": "Interprétation de l'examen"}}
#                 ],
#                 "clinique": [
#                     {{"nom": "Nom de l'examen", "valeur": "Valeur"}}
#                 ]
#             }},
#             "diagnostic": "Diagnostic posé ou laissé vide (chaîne de caractère uniquement)",
#             "traitement": {{
#                 "laboratoire": [
#                     {{"nom": "Nom de l'examen", "observation": "Observation"}}
#                 ],
#                 "imagerie": [
#                     {{"nom": "Nom de l'examen", "observation": "Observation"}}
#                 ],
#                 "ophtalmologie": [],
#                 "medicaments": [
#                     {{"nom": "Nom du médicament", "dose": "Dosage", "posologie": "Posologie", "duree": "Durée"}}
#                 ],
#                 "recommendation": "Où la patiente a été recommandé pour la suite de sa prise en charge",
#                 "autres": [
#                     {{"nom": "Nom", "observation": "Observation"}}
#                 ]
#             }}
#         }}
        
#         **Règles essentielles à respecter** :  
#         **Aucune information hors JSON** : Seul un JSON valide doit être retourné, **sans texte explicatif**.  
#         **Aucune invention ni interprétation** : Si une information est absente, **laisser le champ vide ou ne pas l'inclure**.  
#         **Respect strict de la structure** : Ne pas mélanger les catégories (ex : ne pas mettre des médicaments dans "laboratoire").  
#         **Formatage JSON exact** : Respecter l'ordre et les types de données indiqués.  

#         ### **Détails à extraire** :  
#         - **"anamnèse"** : Uniquement un résumé médical des symptômes et antécédents, **sans recommandation, ni demande d'examen**.  
#         - **"examen"** :  
#           - **"clinique"** : Liste des examens cliniques effectués avec leurs valeurs. **Ne pas inclure des examens qui ne sont pas mentionnés explicitement.**  
#           - **"paraclinique"** : Résultats des examens paracliniques **uniquement si une interprétation du médecin est donnée**. **Sinon, ne pas inclure l'examen.** 
#             Exemple après prise de connaissance des resultats des examens biologiques du patient, le medecin fait savoir que le patient a un niveau de globule blanc peu élevé  suggérant une infection bactérienne. 
#             Voici la sortie attendue pour cette exemple: {{"nom": "Globules blancs",
#     #                                                    "valeur": "Peu élevé, suggérant une infection bactérienne"}}  
#         - **"diagnostic"** : Diagnostique(s) mentionné(s) par le médecin (même sous forme d'hypothèse). format de type chaîne de caractère.  
#         - **"traitement"** :  
#           - **"laboratoire"** : Examens de laboratoire demandés, avec observations associées.  
#           - **"imagerie"** : Examens d'imagerie prescrits.  
#           - **"ophtalmologie"** : Consultations recommandées en ophtalmologie.  
#           - **"medicaments"** : Médicaments prescrits. Doit contenir les champs: nom, dose, posologie, durée. Marquer vide le cas échéant.  
#           - **"recommendation"** : Médecin ou spécialiste recommandé. Marquer comme vide si aucune recommandation vers un medecin n'a été faite.  
#           - **"autres"** : Recommandations ou traitements non classables ailleurs.  

#         **IMPORTANT** :  
#         1. **Aucune reformulation excessive** : Il faut simplement extraire et organiser les informations sans ajouter d'explications.  
#         2. **Aucun élément hors JSON** ne doit être produit.  
#         3. **Remplir uniquement avec les données disponibles** dans la consultation.  
#         """),
#         # few_shot_prompt_consult,
#         ("human", "{input}")
#     ]
# )

prompt_consultation_resume = ChatPromptTemplate.from_messages(
    [
        ("system", """ Vous allez recevoir un transcript d'une consultation entre un médecin et son patient. Votre tâche est d'extraire les informations essentielles
        et de les organiser sous forme d'un fichier JSON structuré. tiens comptes du feminin et du masculin en fonction du sexe du patient (le patient pour un homme et la patiente pour une femme).
        
        **Voici les informations à extraire** :
        "anamnèse" : **L'anamnèse** corresponds aux informations fournis par le patient au médédin lors de la consultation (symptômes, antécedant médicaux, traitement en cours). En aucun
                cas tu ne doit faire mention d'examen lors du récapilatif au niveau de l'anamnèse. Rassure-toi de bien réprendre toutes les informations notament les antécédant
                médicaux, les symptômes, durée, traitement en cours, etc... C'est crucial et important pour la suite. Contente toi juste de faire un résumé qui prend bien en commpte l'ensemble des informations et des réponses aux questions fournis par le patient. Tous les détails importants sur les symptômes et l'état du patient doivent figurer. Rappelle-toi que c'est toi le medecin donc à la lecture de ton 
                résumé cela doit sembler être redigé par le medecin et non par une autre personne.
            Ne mentionnne aucune recomandation ou des phrases du genre :
                - Il est essentiel de procéder à une évaluation approfondie pour déterminer la cause sous-jacente de ces symptômes et élaborer un plan de traitement adapté.
                - Resumé de la consultation 
                - Une évaluation plus approfondie est nécessaire pour déterminer la cause de la douleur au sein gauche et des maux de tête. Des examens complémentaires, 
                    tels qu'une mammographie ou une échographie, pourraient être envisagés pour évaluer la nature de la douleur au sein gauche.
                - Ne mentionne aucune information concernant les resultats ou les examens cliniques, paracliniques, prescriptions. Reste juste sur le résumé de l'anamnèse. Par exemple
                    evite les phrases du genre 'Les examens révèlent un processus inflammatoire avec une CRP et une VS élevées. Le facteur rhumatoïde (AMGFR) est également élevé, et l'anti-CCP est positif, suggérant une polyarthrite rhumatoïde.
                    L'hémogramme montre de légères anomalies inflammatoires, indiquant un début d'atteinte chronique.' car ces infoormations ne doivent pas se trouver dans l'anmèse.
                    Content-toi de juste faire un résumé c'est tout. C'est essentiel de garder cela a l'esprit, c'est crucial de ne fournir que le résume.
        "examen clinique" : **L'examen clinique** corresponds aux examens ou obersavations physiques effectués par le médécin sur le patient. 
            Liste des examens cliniques mentionnés ou effectués avec les champs (nom, valeur). Uniquement les examens cliniques mentionnés ou éffectués. 
            Tu dois lister tous les examens cliniques effectués que le résultat soit normal ou pas car le but est de pouvoir savoir tous les examens cliniques éffectués.
        "examen paraclinique" : **L'examen paraclinique** corresponds aux examens (non physique) liés aux examens effectués par le patient auquels le médécin apporte une observation ou une interpretation.
            Si le medecin a effectué une interpretation des examens paracliniques fournis par le patient alors recueillir la liste des interprétations des examens paracliniques effectués avec les champs (nom, resultat). 
            Uniquement les resultats des examens paracliniques qui ont été interpretés par le medecin. En abscence d'interprétation d'un examen paraclinique celui-ci ne doit pas être pris en compte dans la liste.
        "diagnostic" : Le ou Les diagnostics mentionnés par le médecin.Cela peut être une simple hypothèse ou dit de façon certaines. Marque comme vide le cas échéant.
        "traitement" : Les traitements ou suggestions proposés par le médecin, organisés en plusieurs catégories (Marque comme vide le cas échéant.) :
            "laboratoire" : Les examens de laboratoire demandés (nom et observations associées).
            "Imagerie" : Les examens d'imagerie médicale prescrits (nom et observations associées).
            "Ophtalmologie" : Les recommandations spécifiques pour consulter un ophtalmologue (liste des éléments recommandés).
            "medicaments" : Les médicaments prescrits avec leurs détails (nom, dose, posologie, durée).
            "recommendation" : le médécincin chez qui le patient à été recommendé ou envoyé, laisser vide le cas écheant.
            "autres" : Toute autre recommandation ou traitement proposé (nom et observations associées par exemple ca peut etre le fait qu'un medecin fasse la recommandation vers un autre). 
        
        REMARQUE: Ne confond pas examen paraclinique et prescription. les prescriptions se sont des examens que le medecin a demandé ou des medicament prescrit tandis que les examens paracliques sont des examens déja effectués pas le patient auquel le medecin a fait une interpretation.
         
         ATTENTION: Ton but est d'extraire les informations et de les restituer au format JSON. Tu ne dois ajouter aucune informations en dehors du format JSON. SEUL LE FORMAT JSON DOIT ETRE RENVOYER. Marque comme vide toute information non disponible dans la conversation.

         TRES IMPORTANT : Voici le format JSON attendu : 
        
        {{
            "anamnèse": "Résumé de l'anamnèse",
            "examen":{{ 
                {{"paraclinique" : [
                    {{"nom": [Nom de l'examen],
                    "valeur" : [interpretation de l'examen],}}, ]
                }},
                {{ "clinique": [{{"nom": [Nom de l'examen],
                    "valeur" : [valeur]}},
                    ]               
                }}
            }},
         
            "diagnostic": "diagnostic du medecin",
            "traitement": {{ 
                "laboratoire": [
                    {{"nom": "Hémogramme complet", "observation": "Rechercher des signes d'inflammation"}}
                ],
                "imagerie": [
                    {{"nom": "Radiographie des articulations", "observation": "Rechercher des signes de dégâts articulaires"}}
                ],
                "ophtalmologie": [],
                "medicaments": [
                    {{"nom": "Aspirine", "dose": "100mg", "posologie": "1 comprimé par jour", "duree": "7 jours"}}
                ],
                "recommendation": "vers qui le patient à été recommendé. laisser vide le cas écheant",
                "autres": [{{"nom": "titre de l'observation", "observation": "detail de l'observation"}}]
            }}
        }}
        Contente-toi juste d'extraire les informations et de les ranger dans les sections corresppondantes. Si pour une section donnée l'information n'est pas disponible
        alors laisse cette section vide au lieu d'ajouter des informatons incorrectes. Et aussi une chose très importante: respecte le format JSON attentu.
         
         **NOTE BIEN**:
            pour les champs **valeur** ne mentionne que la valeur: c'est ce qui est important. Pas besoin d'ajouter les commentaires du médécin comme *élevé, plutôt anormal, normal, etc...* ou tout commentaire similaire.
            Si le transcript que tu reçois n'est pas une conversation medical alors revois le format jSon mais avec des champs vide.
        """
    ),
    # few_shot_prompt_consult,
    ("human", "{input}")

    ]
)
# prompt_consultation = ChatPromptTemplate.from_messages(
#     [
#         ("system", """ Vous allez recevoir un transcript d'une consultation entre un médecin et son patient. Votre tâche est d'extraire les informations essentielles
#         et de les organiser sous forme d'un fichier JSON structuré. tiens comptes du feminin et du masculin en fonction du sexe du patient (le patient pour un homme et la patiente pour une femme).
        
#         Voici les informations à extraire :
#         "anamnèse" : Contente toi juste de faire un résumé qui prend bien en commpte l'ensemble des informations et des réponses aux questions fournis par le patient. Tous les détails importants sur les symptômes et l'état du patient doivent figurer. Rappelle-toi que c'est toi le medecin donc à la lecture de ton 
#          résumé cela doit sembler être redigé par le medecin et non par une autre personne.
#          Ne mentionnne aucune recomandation ou des phrases du genre :
#           - Il est essentiel de procéder à une évaluation approfondie pour déterminer la cause sous-jacente de ces symptômes et élaborer un plan de traitement adapté.
#           - Resumé de la consultation 
#           - Une évaluation plus approfondie est nécessaire pour déterminer la cause de la douleur au sein gauche et des maux de tête. Des examens complémentaires, 
#            tels qu'une mammographie ou une échographie, pourraient être envisagés pour évaluer la nature de la douleur au sein gauche.
#           - ne mentionne aucune information concernant les examens cliniques, paracliniques, prescriptions. Reste juste sur le résumé de l'anamnèse.
#             Content-toi de juste faire un résumé c'est tout. C'est essentiel de garder cela a l'esprit, c'est crucial de ne fournir que le résume.
#         "examen clinique" : liste des examens cliniques mentionnés ou effectués avec les champs (nom, valeur). Uniquement les examens cliniques mentionnés ou éffectués. Marque comme vide le cas échéant.
#         "examen paraclinique" : Si le medecin a effectué une analyse des examens paracliniques fournis par le patient alors recueillir la liste des interprétations des examens paracliniques effectués avec les champs (nom, resultat). 
#             Uniquement les resultats des examens paracliniques qui ont été interpretés par le medecin. En abscence d'interprétation d'un examen paraclinique celui-ci ne doit pas être pris en compte dans la liste.
#         "diagnostic" : Le ou Les diagnostics mentionnés par le médecin.Cela peut être une simple hypothèse ou dit de façon certaines. Marque comme vide le cas échéant.
#         "traitement" : Les traitements ou suggestions proposés par le médecin, organisés en plusieurs catégories (Marque comme vide le cas échéant.) :
#             "laboratoire" : Les examens de laboratoire demandés (nom et observations associées).
#             "Imagerie" : Les examens d'imagerie médicale prescrits (nom et observations associées).
#             "Ophtalmologie" : Les recommandations spécifiques pour consulter un ophtalmologue (liste des éléments recommandés).
#             "medicaments" : Les médicaments prescrits avec leurs détails (nom, dose, posologie, durée).
#             "recommendation" : le medecin chez qui le patient à été recommendé, laisser vide le cas écheant.
#             "autres" : Toute autre recommandation ou traitement proposé (nom et observations associées par exemple ca peut etre le fait qu'un medecin fasse la recommandation vers un autre). 
        
         
#          ATTENTION: Ton but est d'extraire les informations et de les restituer au format JSON. Tu ne dois ajouter aucune informations en dehors du format JSON. SEUL LE FORMAT JSON DOIT ETRE RENVOYER. Marque comme vide toute information non disponible dans la conversation.

#          TRES IMPORTANT : Voici le format JSON attendu : 
        
#         {{
#             "anamnèse": "Résumé de l'anamnèse",
#             "examen":{{ 
#                 {{"paraclinique" : [
#                     {{"nom": [Nom de l'examen],
#                     "valeur" : [interpretation de l'examen],}}, ]
#                 }},
#                 {{ "clinique": [{{"nom": [Nom de l'examen],
#                     "valeur" : [valeur]}},
#                     ]               
#                 }}
#             }},
         
#             "diagnostic": "diagnostic du medecin",
#             "traitement": {{ 
#                 "laboratoire": [
#                     {{"nom": "Hémogramme complet", "observation": "Rechercher des signes d'inflammation"}}
#                 ],
#                 "imagerie": [
#                     {{"nom": "Radiographie des articulations", "observation": "Rechercher des signes de dégâts articulaires"}}
#                 ],
#                 "ophtalmologie": [],
#                 "medicaments": [
#                     {{"nom": "Aspirine", "dose": "100mg", "posologie": "1 comprimé par jour", "duree": "7 jours"}}
#                 ],
#                 "recommendation": "vers qui le patient à été recommendé. laisser vide le cas écheant",
#                 "autres": [{{"nom": "titre de l'observation", "observation": "detail de l'observation"}}]
#             }}
#         }}
#         Contente-toi juste d'extraire les informations et de les ranger dans les sections corresppondantes. Si pour une section donnée l'information n'est pas disponible
#         alors laisse cette section vide au lieu d'ajouter des informatons incorrectes. Et aussi une chose très importante: respecte le format JSON attentu.
#         """
#     ),
#     ("human", "{input}")

#     ]
# )
prompt_format = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu es l'assistant d'un medecin expert en modification de texte. Modifie le texte suivant {input} en te servant des cosignes suivantes : {instruction}. \
         Contente toi juste de modifier le texte et rien d'autre sans omettre ou rajouter de details inutiles. Eventuellement tu peux corriger les fautes d'orthographes ou de grammaires.\
         Tu ne dois pas renvoyer une liste de texte ou un json ni du mardown mais juste le texte que tu as modifier sous forme de string ou chaine caractère."),
        ("user", "{input}")
    ]
)

prompt_response_to_json = ChatPromptTemplate.from_messages(
    [
        ("system", """transforme moi le texte de l'utlisateur en format json bien structuré."""),
        ("user", "{input}")
    ]
)



# format texte (voice editor)
examples = [
    {
        "input": "Le patient présente une série de symptômes qui se manifestent principalement la nuit. \
            Il ressent des douleurs, de la fièvre, des douleurs articulaires et des crampes abdominales. \
            En outre, il éprouve des maux de tête violents. Le patient est incapable de déterminer la cause de ces symptômes. \
            Une évaluation plus approfondie sera nécessaire pour déterminer la cause sous-jacente de ces symptômes et élaborer un plan de traitement approprié.",

        "instruction": "Va dans le test. Au lieu de dire que le patient est incapable de déterminer la cause de ses symptômes,\
             dis plutôt que la cause de ses symptômes reste à déterminer pour le moment.",
        
        "output": "Le patient présente une série de symptômes qui se manifestent principalement la nuit. \
            Il ressent des douleurs, de la fièvre, des douleurs articulaires et des crampes abdominales. \
            En outre, il éprouve des maux de tête violents. la cause de ses symptômes reste à déterminer pour le moment. \
            Une évaluation plus approfondie sera nécessaire pour déterminer la cause sous-jacente de ces symptômes et élaborer un plan de traitement approprié."
    },

    {
        "input": "Le patient, un homme, présente des symptômes de douleurs, de crampes et d'envies de vomir depuis quelques jours.\
            Il ressent également une fatigue générale et des maux de tête. Le patient n'a pas d'idée sur l'origine de ces symptômes. \
            Il est important de procéder à une évaluation plus approfondie pour déterminer la cause sous-jacente de ces symptômes. \
            Des examens complémentaires et une anamnèse plus détaillée seront nécessaires pour établir un diagnostic et un plan de traitement adapté.",
        
        "instruction": "Va dans le test, au lieu de dire que l'homme présente des douleurs, des crampes et d'envie de vomir depuis quelques jours, \
            dis plutôt qu'il présente ces envies-là depuis 5 jours, donc il présente des douleurs, des crampes et d'envie de vomir depuis 5 jours.",

        "output": "Le patient, un homme, présente des symptômes de douleurs, de crampes et d'envies de vomir depuis 5 jours.\
            Il ressent également une fatigue générale et des maux de tête. Le patient n'a pas d'idée sur l'origine de ces symptômes. \
            Il est important de procéder à une évaluation plus approfondie pour déterminer la cause sous-jacente de ces symptômes. \
            Des examens complémentaires et une anamnèse plus détaillée seront nécessaires pour établir un diagnostic et un plan de traitement adapté.",
    },

    {
        "input": "Traitement recommandé \
            Pour traiter les symptômes présentés par le patient, je recommande les traitements suivants : \
                Médicament : Prednisone 10 mg/jour pendant au moins 2 semaines.\
                Justification : La prednisone est un corticostéroïde qui peut aider à réduire l'inflammation et les symptômes cutanés. \
                    Elle est également utile pour lutter contre la fatigue et les douleurs articulaires en raison de son effet anti-épileptique et anti-inflammatoire.\
                Médicament : Ibuuproène 400 mg t.t.d. pendant au moins 2 semaines.\
                Justification : L'ibuprofène est un non-stéroïde anti-inflammatoire (NSAID) qui peut aider à réduire les douleurs articulaires et les maux de tête en raison de sa capacité anti-inflammatoire.\
                     Cela permet également une meilleure absorption des vitamines.\
                Suppléments : Vitamine D 2000 UI/jour et vitamine C 1000 mg/jour pendant au moins 2 semaines.\
                Justification : Les suppléments de vitamine D peuvent aider à améliorer les symptômes cutanés en raison de leur rôle anti-inflammatoire. \
                    La vitamine C est également utile pour lutter contre la fatigue et les maux de tête.\
                Dietétique : Je recommande une alimentation équilibrée riche en fruits, légumes, céréales complètes et protéines pour aider à contrôler le stress et améliorer la santé générale du patient.\
            Examen médical obligatoires \
            Je recommande les examens suivants : \
            Radiologie du genoux et de l'ostéodystrophie \
            Laboratoire complet (sang, urine) pour rechercher tout trouble métabolique ou infection \
            Suivi médical \
            Il est essentiel que le patient revienne à la consultation après 2 semaines pour évaluer les résultats du traitement et ajuster le plan de traitement si nécessaire.",

        "instruction": "Vas dans le médicament proposé là, le premier médicament, \
            on a mis une dose de 10 mg par jour, pendant au moins deux semaines, tu mets plutôt paracétamol sur la même durée",
        
        "output": "Traitement recommandé \
            Pour traiter les symptômes présentés par le patient, je recommande les traitements suivants : \
                Médicament : Paracétamol 10 mg/jour pendant au moins 2 semaines.\
                Justification : La prednisone est un corticostéroïde qui peut aider à réduire l'inflammation et les symptômes cutanés. \
                    Elle est également utile pour lutter contre la fatigue et les douleurs articulaires en raison de son effet anti-épileptique et anti-inflammatoire.\
                Médicament : Ibuuproène 400 mg t.t.d. pendant au moins 2 semaines.\
                Justification : L'ibuprofène est un non-stéroïde anti-inflammatoire (NSAID) qui peut aider à réduire les douleurs articulaires et les maux de tête en raison de sa capacité anti-inflammatoire.\
                     Cela permet également une meilleure absorption des vitamines.\
                Suppléments : Vitamine D 2000 UI/jour et vitamine C 1000 mg/jour pendant au moins 2 semaines.\
                Justification : Les suppléments de vitamine D peuvent aider à améliorer les symptômes cutanés en raison de leur rôle anti-inflammatoire. \
                    La vitamine C est également utile pour lutter contre la fatigue et les maux de tête.\
                Dietétique : Je recommande une alimentation équilibrée riche en fruits, légumes, céréales complètes et protéines pour aider à contrôler le stress et améliorer la santé générale du patient.\
            Examen médical obligatoires \
            Je recommande les examens suivants : \
            Radiologie du genoux et de l'ostéodystrophie \
            Laboratoire complet (sang, urine) pour rechercher tout trouble métabolique ou infection \
            Suivi médical \
            Il est essentiel que le patient revienne à la consultation après 2 semaines pour évaluer les résultats du traitement et ajuster le plan de traitement si nécessaire."
    },

    {
        "input": "Le patient, un homme, présente des symptômes de douleurs, de crampes et d&#39;envies de vomir depuis 5 jours.\
                    Il ressent également une fatigue générale et des maux de tête. Le patient n&#39;a pas d&#39;idée sur l&#39;origine de ces symptômes. \
                    Il est important de procéder à une évaluation plus approfondie pour déterminer la cause sous-jacente de ces symptômes.\
                    Des examens complémentaires et une anamnèse plus détaillée seront nécessaires pour établir un diagnostic et un plan de traitement adapté",

        "instruction": "Modifie 5 joues, pas 7 joues",

        "output" : "Le patient, un homme, présente des symptômes de douleurs, de crampes et des envies de vomir depuis 7 jours.\
                    Il ressent également une fatigue générale et des maux de tête. Le patient n&#39;a pas d&#39;idée sur l&#39;origine de ces symptômes. \
                    Il est important de procéder à une évaluation plus approfondie pour déterminer la cause sous-jacente de ces symptômes.\
                    Des examens complémentaires et une anamnèse plus détaillée seront nécessaires pour établir un diagnostic et un plan de traitement adapté"
    },
    {
        "input":"Opacité au niveau de l'inférieure droit",
        "instruction":"Il s'agit d'une opacité au niveau du lobe inférieur droit.",
        "output":"Opacité au niveau du lobe inférieure droit"
    }

]
# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\n{instruction}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
# print(few_shot_prompt.invoke({}).to_messages())
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """Tu es un assistant expert en modification de textes médicaux.

            Ta mission : Modifier uniquement le texte fourni selon l'instruction donnée.

            Règles strictes :
                - Ne retourne que le texte modifié, sans ajout, pas d'explication ni formatage.
                - Corrige uniquement l'orthographe et la grammaire si nécessaire.
                - Respecte scrupuleusement l'instruction. 

            Si l'instruction est ambiguë :
                - Tente de la comprendre en la recontextualisant avec le texte fourni.
                - Si elle reste floue, c'est-a dire si l'insctruction n'est pas claire alors retourne le texte initial sans rien modifier: c'est primodial.
            NOTE BIEN : N'AJOUTE PAS DE CARACTERE SPECIAUX QUI NE FIGURE PAS DANS LE TEXTE A MODIFIER. EVITE 'AJOUTER LES TRUCS DU GENRE "```\n" AU DEBUT ET  "\n```" A LA FIN.
                RESPECTE JUSTE L'INSTRUCTION ET SUIVANT LA LOGIQUE DU FORMATAGE MAIS SANS RIEN AJOUTER DE PLUS. SI L4INSTRUCTION N'EST PAS COMPREHENSIBLE RENVOI LE TEXTE INITIAL SANS RIEN AJOUTER.
         """),
         few_shot_prompt,
        ("human", 
         "Voici le texte à modifier : \n**{input}** \
         Et voici les consignes de modification : \n**{instruction}**")
    ]
)
prompt_paraclinique = ChatPromptTemplate.from_messages(
    [
         ("system", """
            Tu es un assistant médical expert en médecine. Ton rôle est d'aider les médecins à identifier les examens paracliniques pertinents à effectuer pour confirmer un diagnostic, 
        préciser une pathologie, ou surveiller l'évolution d'une condition médicale

          Sur la base de des informations fournis, propose une liste d'examens paracliniques à réaliser. 
          Justifie chaque examen en expliquant son utilité dans le contexte clinique décrit. Reste rigoureux et adapté aux besoins du médecin.

          En vous basant sur les informations fournis :
            1. Listez les examens paracliniques que le médecin pourrait effectuer immédiatement.
            2. Expliquez brièvement pourquoi chaque examen paraclinique est pertinent pour l'évaluation des symptômes décrits.
            3. Assurez-vous que vos suggestions sont adaptées aux informations données et évitez les examens inutiles ou excessifs.

            Format attendu de la réponse (Obligatoirement toujours au format JSON c'est tres Important pour la suite). exemple :
        {{"Paraclinique" : [{{[Nom de l'examen],
            "But" : [Raison pour laquelle cet examen est recommandé],}}, ],
            "Suggestion": [Si les informations sont insuffisantes, indiquez les questions supplémentaires à poser au patient pour affiner vos suggestions]
        }}
              
        """),
        ("human", "Voici le résume de la conversation : \n{input}")
    ]
)
prompt_clinique = ChatPromptTemplate.from_messages(
    [
        ("system", """
            Vous êtes un assistant médical spécialisé dans l'aide au diagnostic clinique. Votre rôle est de suggérer des examens cliniques pertinents en fonction des informations fournies.

            En vous basant sur es informations fournis :
            1. Listez les examens cliniques que le médecin pourrait effectuer immédiatement (examen visuel, palpation, auscultation, etc.).
            2. Expliquez brièvement pourquoi chaque examen clinique est pertinent pour l'évaluation des symptômes décrits.
            3. Assurez-vous que vos suggestions sont adaptées aux informations données et évitez les examens inutiles ou excessifs.

            Format attendu de la réponse (Obligatoirement toujours au format JSON c'est tres Important pour la suite). exemple :
        {{["Clinique" : 
           {{ "Nom": [Nom de l'examen],
            "But" : [Raison pour laquelle cet examen est recommandé],
            "Méthode" : [Courte explication de comment l'examen est effectué],}}, ]
            "Suggestion": [Si les informations sont insuffisantes, indiquez les questions supplémentaires à poser au patient pour affiner vos suggestions]
        }}
              
        """),
        ("human", "Voici le résume de la conversation : \n{input}")
    ]
)



# reg flag
example_regflag = [
    {
        "input": "Le patient, 73 ans, insuffisant rénal chronique (DFG = 25 ml/min). Il est suivi pour un cancer du rein et doit subir un bilan d'extension.",
        "prescription_medecin": "  'examen': 'Scanner thoraco-abdomino-pelvien avec injection', \
            'indication': 'Bilan d'extension du cancer', \
            'date': '2024-05-20'",
        "alerte_assistant": "Le patient présente une insuffisance rénale sévère (DFG = 25 ml/min). L'injection de produit de contraste iodé peut aggraver sa fonction rénale. Vérifiez la nécessité de l'examen et envisagez une IRM avec gadolinium si possible."
    },
    {
        "input": "Le patient, 68 ans, porteur d'un pacemaker depuis 2019 en raison d'une bradycardie sévère. Il présente des symptômes neurologiques (troubles de la mémoire).",
        "prescription_medecin": {
            "examen": "IRM cérébrale",
            "indication": "Bilan de troubles neurologiques",
            "date": "2024-04-12"
        },
        "alerte_assistant": "Le patient est porteur d'un pacemaker non compatible avec l'IRM, ce qui représente un risque vital (dysfonctionnement, arrêt du dispositif). Vérifiez la compatibilité ou envisagez un autre examen (scanner cérébral)."
    },
    {
        "input": "La patiente, 30 ans, enceinte de 14 semaines, hospitalisée pour douleurs abdominales aiguës. Aucune mention de contre-indications ou d'antécédents médicaux majeurs.",
        "prescription_medecin": {
            "examen": "Scanner abdominal",
            "indication": "Douleurs abdominales aiguës",
            "date": "2024-03-05"
        },
        "alerte_assistant": "Le scanner abdominal est contre-indiqué en cas de grossesse en raison des risques de malformations fœtales dues aux radiations ionisantes. Envisagez une échographie ou une IRM selon l'indication clinique."
    },
    {
        "input": "Le patient, 48 ans, souffrant de douleurs lombaires chroniques depuis plusieurs années. Il a déjà eu 5 radiographies lombaires au cours des 12 derniers mois.",
        "prescription_medecin": {
            "examen": "Radiographie du rachis lombaire",
            "indication": "Douleurs lombaires chroniques",
            "date": "2024-02-10"
        },
        "alerte_assistant": "Le patient a déjà subi 5 radiographies lombaires en un an. Une exposition excessive aux rayons X peut entraîner des risques à long terme. Envisagez une IRM ou un traitement conservateur avant de refaire une radiographie inutile."
    },
    {
        "input": "Le patient, 52 ans, hospitalisé pour suspicion d'embolie pulmonaire. Il a déjà eu 3 scanners thoraciques cette année, dont 2 en une semaine pour suivre l'évolution de son état.",
        "prescription_medecin": {
            "examen": "Scanner thoracique",
            "indication": "Suivi de l'embolie pulmonaire",
            "date": "2024-01-15"
        },
        "alerte_assistant": "Le patient a déjà reçu 3 scanners thoraciques cette année. La dose annuelle maximale de rayons X est en passe d'être dépassée, augmentant le risque de cancers induits par les radiations. Vérifiez si une alternative (échographie Doppler, IRM) est possible."
    },
    {
        "input": "La patiente, 31 ans, enceinte de 12 semaines, souffrant d'acné sévère depuis plusieurs années. Aucun autre antécédent médical.",
        "prescription_medecin": {
            "medicament": "Isotrétinoïne",
            "dose": "20 mg par jour",
            "duree": "3 mois",
            "voie": "Orale"
        },
        "alerte_assistant": "L'Isotrétinoïne est STRICTEMENT contre-indiquée en cas de grossesse. Ce médicament est tératogène et entraîne de graves malformations fœtales. Vérifiez immédiatement la prescription et envisagez d'autres options après évaluation dermatologique."
    },
    {
        "input": "La patiente, 29 ans, sans antécédents médicaux particuliers. Elle consulte pour une angine fébrile avec test de détection rapide positif au streptocoque A.",
        "prescription_medecin": {
            "medicament": "Amoxicilline",
            "dose": "1 g toutes les 8 heures",
            "duree": "4 semaines",
            "voie": "Orale"
        },
        "alerte_assistant": "La durée de traitement par Amoxicilline pour une angine streptococcique est de 7 à 10 jours. Une antibiothérapie prolongée augmente le risque d'effets secondaires et de résistances bactériennes. Vérifiez la durée de prescription."
    },
    {
        "input": "Le patient, 55 ans, hospitalisé pour une intervention chirurgicale sous anesthésie générale. Aucun antécédent médical significatif.",
        "prescription_medecin": {
            "medicament": "Propofol",
            "dose": "200 mg",
            "duree": "Dose unique",
            "voie": "Orale"
        },
        "alerte_assistant": "Le Propofol est un anesthésique général administré UNIQUEMENT par voie intraveineuse. La voie orale n'est pas adaptée et pourrait être inefficace ou dangereuse. Vérifiez la prescription immédiatement."
    },
    {
        "input": "La patiente, 45 ans, souffre de dépression et est sous Fluoxétine (Prozac) depuis 6 mois. Elle n'a aucun antécédent de convulsions ou de troubles neurologiques.",
        "prescription_medecin": {
            "medicament": "Tramadol",
            "dose": "100 mg toutes les 6 heures",
            "duree": "5 jours",
            "voie": "Orale"
        },
        "alerte_assistant": "Le patient est sous Fluoxétine, un antidépresseur inhibiteur de la recapture de la sérotonine. L'ajout de Tramadol augmente fortement le risque de syndrome sérotoninergique (agitation, tremblements, hyperthermie, coma). Veuillez envisager un autre analgésique (ex. paracétamol, anti-inflammatoires non stéroïdiens si non contre-indiqués)."
    },
    {
        "input": "Le patient, 64 ans, atteint de goutte chronique. Il a déjà eu plusieurs crises et a été traité par Colchicine par le passé sans effets secondaires majeurs. Il ne prend actuellement aucun autre médicament.",
        "prescription_medecin": {
            "medicament": "Colchicine",
            "dose": "10 mg par jour",
            "duree": "7 jours",
            "voie": "Orale"
        },
        "alerte_assistant": "La dose prescrite de Colchicine (10 mg/jour) est toxique et peut entraîner une insuffisance multi-organique. La dose recommandée est de 1 mg suivi de 0.5 mg après 1h, puis 0.5 mg 1 à 2 fois par jour. Veuillez corriger immédiatement la prescription."
    },
    {
        "input": "La patiente, 55 ans, souffre d'hypercholestérolémie. Son dernier bilan lipidique montre un taux de LDL élevé. Elle est actuellement sous traitement par Atorvastatine 20mg.",
        "prescription_medecin": "Simvastatine 40mg",
        "alerte_assistant": "La patiente prend déjà une statine (Atorvastatine). L'ajout de Simvastatine augmente le risque d'effets secondaires (myopathie, rhabdomyolyse). Vérifiez si un ajustement de la dose de l'Atorvastatine est préférable."
    },
    {
        "input": "Le patient, 58 ans, souffre d'une insuffisance rénale chronique avancée (stade 4). Son dernier bilan indique un débit de filtration glomérulaire (DFG) de 18 mL/min. Elle est sous traitement par Ramipril pour la gestion de son hypertension.",
        "prescription_medecin": "Ibuprofène 400mg",
        "alerte_assistant": "Le patient souffre d'insuffisance rénale sévère (DFG = 18 mL/min). L'Ibuprofène est contre-indiqué car il peut aggraver la fonction rénale. Veuillez prescrire un antalgique plus sûr (paracétamol par exemple)."
    },
    {
        "input": "Le patient, 50 ans, diabétique de type 2, sous traitement par Metformine. Il a passé un scanner thoracique il y a trois jours, qui n'a révélé aucune anomalie.",
        "prescription_medecin": "Scanner thoracique",
        "alerte_assistant": "Le patient a déjà passé un scanner thoracique il y a 3 jours. Une exposition répétée aux rayons X peut être nocive. Vérifiez si cet examen est vraiment nécessaire ou s'il existe une alternative (IRM, échographie)."
    },
    {
        "input": "La Patiente, 32 ans, a une allergie confirmée à la Pénicilline. Aucun autre antécédent médical majeur n'est rapporté. Il ne prend actuellement aucun médicament.",
        "prescription_medecin": "Amoxicilline",
        "alerte_assistant": "La patiente est allergique à la Pénicilline. L'administration d'Amoxicilline peut provoquer une réaction allergique sévère (choc anaphylactique). Veuillez prescrire une alternative non bêta-lactamine."
    },
    {
        "input": "La patiente, 67 ans, souffre d'hypertension et a fait un AVC en 2018. Elle prend actuellement de la Warfarine pour éviter les caillots sanguins. Son dernier bilan montre un INR à 2.8, indiquant un contrôle correct de son anticoagulation.",
        "prescription_medecin": "Aspirine 100mg",
        "alerte_assistant": "La patiente prend déjà de la Warfarine. L'ajout d'Aspirine augmente le risque d'hémorragie sévère. Veuillez envisager une alternative ou surveiller attentivement le patient."
    },
    {
        "input": "Le patient, 67 ans, a subi une pose de stents métalliques il y a 5 ans après un infarctus. Il est sous anticoagulants (Xarelto) pour prévenir les thromboses. Il souffre actuellement de douleurs cervicales chroniques.",
        "prescription_medecin": {
            "examen": "IRM cervicale",
            "médicaments": ["Xarelto (rivaroxaban)"],
            "indication": "Douleurs cervicales chroniques",
            "date": "2024-07-10"
        },
        "alerte_assistant": "Le patient est sous anticoagulants et possède des stents métalliques. L'IRM peut être dangereuse en raison des interactions avec le métal implanté et du risque accru d'hémorragie. Vérifiez la compatibilité des stents et la nécessité absolue de l'examen. Un scanner pourrait être une alternative."
    },
    {
        "input": "Le patient, 74 ans, diabétique de type 2 sous Metformine, avec une insuffisance rénale modérée (DFG = 40 ml/min). Il doit passer un scanner thoracique avec injection de produit iodé pour un bilan post-opératoire.",
        "prescription_medecin": {
            "examen": "Scanner thoracique avec injection de produit iodé",
            "médicaments": ["Metformine"],
            "indication": "Bilan post-opératoire",
            "date": "2024-08-02"
        },
        "alerte_assistant": "Le patient est sous metformine et présente une insuffisance rénale modérée. L'injection de produit de contraste iodé peut déclencher une acidose lactique sévère. Un ajustement du traitement et une hydratation préventive sont nécessaires avant l'examen."
    },
    {
        "input": "Marie Fontaine, 32 ans, enceinte de 18 semaines, consulte pour des douleurs lombaires intenses. Elle a déjà été traitée par ibuprofène et AINS.",
        "prescription_medecin": {
            "examen": "Scanner lombaire",
            "médicaments": ["Ibuprofène"],
            "indication": "Douleurs lombaires",
            "date": "2024-09-05"
        },
        "alerte_assistant": "Le patient est enceinte de 18 semaines. Les AINS comme l'ibuprofène sont contre-indiqués pendant la grossesse en raison du risque de toxicité fœtale. De plus, un scanner lombaire expose le fœtus aux radiations. Optez pour un traitement non médicamenteux et une IRM si nécessaire."
    },
    {
        "input": "Lucas Moreau, 6 ans, traité pour une infection urinaire sévère avec aminosides (Gentamicine). Il présente des douleurs abdominales persistantes.",
        "prescription_medecin": {
            "examen": "Scanner abdominal",
            "médicaments": ["Gentamicine"],
            "indication": "Douleurs abdominales persistantes",
            "date": "2024-08-12"
        },
        "alerte_assistant": "Le patient est sous gentamicine, un antibiotique néphrotoxique. Un scanner abdominal pourrait aggraver une éventuelle atteinte rénale. De plus, l'exposition aux rayons X est élevée chez l'enfant. Une échographie abdominale pourrait être une alternative plus sûre."
    },
    {
        "input": "La patiente, 5 ans, a subi 4 scanners crâniens en 6 mois suite à des traumatismes crâniens répétés liés à des chutes.",
        "prescription_medecin": {
            "examen": "Scanner crânien",
            "indication": "Nouvelle chute, suspicion de traumatisme crânien",
            "date": "2024-06-10",
        },
        "alerte_assistant": "La patiente a subi 4 scanners crâniens en 6 mois. Chez l'enfant, une exposition excessive aux rayons X augmente le risque de cancers à long terme. Vérifiez si une observation clinique ou une échographie transfontanellaire (si possible) est une alternative.",
    },
    {
        "input": "Homme de 45 ans, sans antécédents médicaux notables. Pas d'allergies connues. Traitement actuel : Paracétamol 1g en cas de douleur. Aucune pathologie chronique.",
        "prescription_medecin": {
            "medicaments": [
                {
                    "nom": "Ibuprofène",
                    "dosage": "400 mg",
                    "frequence": "3 fois par jour",
                    "duree": "5 jours"
                }
            ],
            "examens": [
                {
                    "nom": "Radio du genou",
                    "raison": "Douleur articulaire persistante",
                    "frequence": "1 seule fois"
                }
            ]
        },
        "alerte_assistant": "null"
    },
    {
        "input": "Enfant de 6 ans, en bonne santé générale. Aucun antécédent médical. Pas d'allergies connues.",
        "prescription_medecin": {
            "medicaments": [
                {
                    "nom": "Amoxicilline",
                    "dosage": "250 mg",
                    "frequence": "2 fois par jour",
                    "duree": "7 jours"
                }
            ],
            "examens": []
        },
        "alerte_assistant": "null"
    },
    {
        "input": "Femme de 78 ans, avec hypertension traitée par Amlodipine 5 mg/jour. Pas d'autres pathologies graves. Aucune allergie connue.",
        "prescription_medecin": {
            "medicaments": [
                {
                    "nom": "Paracétamol",
                    "dosage": "500 mg",
                    "frequence": "3 fois par jour",
                    "duree": "7 jours"
                }
            ],
            "examens": [
                {
                    "nom": "Échographie abdominale",
                    "raison": "Douleurs abdominales",
                    "frequence": "1 seule fois"
                }
            ]
        },
        "alerte_assistant": "null"
    },

]
example_regflag_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\n{prescription_medecin}"),
        ("ai", "{alerte_assistant}"),
    ]
)
few_shot_regflag_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_regflag_prompt,
    examples=example_regflag,
)
regflag_final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Tu es un assistant médical intelligent conçu pour analyser les prescriptions des médecins en fonction du dossier médical du patient.

        **OBJECTIF** 
            Ton objectif est de détecter toute incohérence, interaction médicamenteuse, dangereuse, contre-indication, ou exposition excessive aux examens radiologiques, etc... 
            Prends en compte les antécédents médicaux, les traitements en cours, les pathologies, l'âge et les particularités du patient (grossesse, insuffisance rénale, enfant, etc.). 
                - Si une prescription présente un risque, génère une alerte sous forme d'un message structuré expliquant clairement le danger et suggérant une alternative si possible. 
                - Si aucune alerte n'est nécessaire, renvoie **"null"** sans rien ajouter d'autre comme caractère. Sois précis, factuel et rigoureux dans ton analyse tout en évitant de faire les longs discours ou répétant les informations déja disponibles.

        ATTENTION
            Si aucune alerte n'est  nécessaire alors renvoie '*null*': c'est très important au risque de nuir à la suite du processus. 

        **FORMAT ATTENDU**: 
        tu dois absolument respecter le format suivant
            - **alerte**: 'explication du motif de l'alerte'
            - **suggestion**: 'suggestion ou solution pour palier à l'alerte'    
        """),
        few_shot_regflag_prompt,
        ("human", 
         "Voici un résumé du dossier patient : {input} \
         Et voici la prescription du médecin : {prescription_medecin}")
    ]
)



# resume consultation
example_resume_consultation = [
    {
        "input": """Anamnèse : Le patient présente une fièvre modérée (38,5°C) depuis 2 jours, accompagnée de frissons, de fatigue et de maux de tête. 
            Il signale également un écoulement nasal clair et une légère toux sèche.
            Examen clinique :
                Température : 38,5°C
                Fréquence cardiaque : 85 bpm
                Tension artérielle : 120/80 mmHg
                Auscultation pulmonaire : RAS
            Examens paracliniques : Aucun examen paraclinique n'a été réalisé.
            Prescription :
                Paracétamol 1g toutes les 8 heures pendant 3 jours
                Hydratation abondante
                Repos recommandé
            Antécédents médicaux : Aucun antécédent médical notable.
            Traitement en cours : Aucun traitement en cours.
            Infos :
                Âge : 25 ans
                Taille : 1m75
                Poids : 68 kg
                Allergies : Aucune
        """,
        "resume": """Un patient de 25 ans, sans antécédents médicaux ni traitement en cours, consulte pour une fièvre modérée à 38,5°C évoluant depuis 2 jours, accompagnée de fatigue, 
            frissons, écoulement nasal et toux sèche. L'examen clinique est rassurant, sans anomalie pulmonaire. Aucun examen paraclinique n'a été réalisé. Un diagnostic de grippe 
            saisonnière est posé, et un traitement symptomatique à base de paracétamol, repos et hydratation abondante est prescrit.
        """
    },
    {
        "input":"""
            Anamnèse : Le patient ressent une fatigue persistante et une soif excessive depuis une semaine. 
            Il signale également des envies fréquentes d'uriner et une légère vision floue.
            Examen clinique :
                Glycémie capillaire : 2,1 g/L
                Fréquence cardiaque : 90 bpm
                Tension artérielle : 130/85 mmHg
            Examens paracliniques :
                HbA1c : 8,2%
            Prescription :
                Ajustement du traitement par Metformine (augmentation de la dose à 1000 mg matin et soir)
                Régime alimentaire adapté (diminution des glucides rapides)
                Surveillance glycémique quotidienne
            Antécédents médicaux : Diabète de type 2 diagnostiqué il y a 5 ans.
            Traitement en cours : Metformine 850 mg matin et soir.
            Infos :
                Âge : 55 ans
                Taille : 1m70
                Poids : 82 kg
                Allergies : Aucune
        """,
        "resume": """
            Un patient de 55 ans, diabétique de type 2 sous Metformine, consulte pour une fatigue persistante, une soif excessive et des mictions fréquentes depuis une semaine. 
            L'examen clinique montre une glycémie capillaire élevée (2,1 g/L) et une HbA1c à 8,2 %, traduisant un déséquilibre glycémique. 
            Le traitement est ajusté avec une augmentation de la dose de Metformine, des conseils diététiques et une surveillance glycémique renforcée.
        """
    },
    {
        "input": """
            Anamnèse : La patiente signale des brûlures mictionnelles et une sensation de pesanteur pelvienne depuis 3 jours. 
            Elle mentionne une légère fièvre (37,8°C) et une augmentation de la fréquence des mictions.
            Examen clinique :
                Température : 37,8°C
                Douleur à la palpation hypogastrique : présente
                Bandelette urinaire : présence de leucocytes et de nitrites
            Examens paracliniques :
                ECBU : infection à Escherichia coli, antibiogramme en attente
            Prescription :
                Antibiothérapie empirique : Fosfomycine 3g en prise unique
                Hydratation abondante
                Consultation de suivi après antibiogramme
            Antécédents médicaux : Aucune infection urinaire antérieure.
            Traitement en cours : Aucun traitement en cours.
            Infos :
                Âge : 30 ans
                Taille : 1m65
                Poids : 60 kg
                Allergies : Pénicilline
        """,
        "resume": """Une patiente de 30 ans, allergique à la pénicilline, consulte pour des brûlures mictionnelles, une pesanteur pelvienne et une fièvre légère (37,8°C) évoluant depuis 3 jours. 
            L'examen clinique et une bandelette urinaire orientent vers une infection urinaire, confirmée par un ECBU révélant une infection à Escherichia coli. 
            Une antibiothérapie par Fosfomycine en prise unique est instaurée en attendant l'antibiogramme, avec une recommandation d'hydratation abondante et un suivi.
        
        """
    },
    {
        "input":"""Le patient se plaint de maux de tête et de vertiges apparus progressivement depuis une semaine, sans autre symptôme associé.
            Examen clinique :
                Tension artérielle : 160/95 mmHg
                Fréquence cardiaque : 88 bpm
            Examens paracliniques : Aucun examen paraclinique réalisé.
            Prescription :
                Augmentation de la dose d'Amlodipine à 10 mg par jour
                Surveillance tensionnelle quotidienne
                Régime hyposodé
                Consultation de suivi dans un mois
            Antécédents médicaux : Hypertension artérielle diagnostiquée il y a 5 ans.
            Traitement en cours : Amlodipine 5 mg par jour.
            Infos :
                Âge : 60 ans
                Taille : 1m72
                Poids : 85 kg
                Allergies : Aucune
        """,
        "resume":"""Un patient de 60 ans, suivi pour une hypertension artérielle sous Amlodipine 5 mg, consulte pour des maux de tête et des vertiges apparus depuis une semaine. 
            L'examen clinique révèle une tension artérielle élevée à 160/95 mmHg et une fréquence cardiaque de 88 bpm. 
            Un bilan biologique est demandé, et la dose d'Amlodipine est augmentée à 10 mg par jour. Une surveillance tensionnelle régulière et un régime hyposodé sont recommandés.
        """
    },
    {
        "input":"""Anamnèse : Le patient signale une recrudescence des crises d'asthme depuis 2 semaines avec des réveils nocturnes liés à une gêne respiratoire.
            Examen clinique :
                Auscultation pulmonaire : sibilants diffus
                Peak-flow : 250 L/min
            Examens paracliniques : Aucun.
            Prescription :
                Ajout d'un traitement de fond par corticostéroïdes inhalés
                Maintien du Salbutamol en cas de crise
                Consultation de suivi dans un mois
            Antécédents médicaux : Asthme diagnostiqué à l'âge de 12 ans.
            Traitement en cours : Salbutamol inhalé en cas de crise.
            Infos :
                Âge : 35 ans
                Taille : 1m80
                Poids : 75 kg
                Allergies : Aucune
        """,
        "resume": """Un patient de 35 ans, asthmatique sous Salbutamol en inhalation, consulte pour une augmentation de la fréquence des crises avec une gêne respiratoire nocturne. 
            L'auscultation pulmonaire révèle des sibilants diffus. Un peak-flow réduit à 250 L/min confirme un mauvais contrôle de l'asthme. 
            Un traitement de fond par Corticostéroïdes inhalés est ajouté, et une consultation de suivi est programmée.
        """
    },
    {
        "input": """Anamnèse : Le patient consulte pour une fatigue persistante depuis plusieurs semaines, sans autres symptômes.
            Examen clinique :
                Tension artérielle : 140/85 mmHg
            Examens paracliniques :
                Créatinine : 150 µmol/L
                Clairance rénale estimée : 50 mL/min
            Prescription :
                Adaptation du traitement antihypertenseur
                Surveillance néphrologique
                Conseils diététiques
            Antécédents médicaux : Hypertension artérielle.
            Traitement en cours : IEC (Inhibiteurs de l'enzyme de conversion).
            Infos :
                Âge : 50 ans
                Taille : 1m75
                Poids : 80 kg
                Allergies : Aucune
        """,
        "resume":"""Un patient de 50 ans, hypertendu sous traitement, consulte pour une fatigue inexpliquée. 
            L'examen clinique est sans particularité, mais un bilan sanguin révèle une créatinine élevée à 150 µmol/L et une clairance estimée à 50 mL/min. Une insuffisance rénale chronique modérée est suspectée. 
            Un suivi néphrologique est recommandé, avec une adaptation du traitement antihypertenseur et des conseils diététiques visant à limiter la progression de la maladie.
        """
    },
    {
        "input": """Anamnèse : L'enfant a de la fièvre depuis 2 jours et se plaint de douleurs à l'oreille droite.
            Examen clinique :
                Température : 39°C
                Tympan droit inflammatoire et bombé
            Examens paracliniques : Aucun.
            Prescription :
                Amoxicilline pendant 7 jours
                Antipyrétique
            Antécédents médicaux : Aucun.
            Traitement en cours : Aucun.
            Infos :
                Âge : 5 ans
                Taille : 1m05
                Poids : 18 kg
                Allergies : Aucune
        """,
        "resume":"""Un enfant de 5 ans, sans antécédents médicaux, est amené en consultation pour une fièvre à 39°C, une otalgie droite et une irritabilité depuis 2 jours. 
            L'examen ORL montre un tympan droit inflammatoire et bombé, évoquant une otite moyenne aiguë. 
            Une antibiothérapie par Amoxicilline est prescrite pour 7 jours, associée à un antipyrétique pour soulager la fièvre.
        """
    }
]

example_resume_consultation_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{resume}"),
    ]
)

few_shot_resume_consultation_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_resume_consultation_prompt,
    examples=example_resume_consultation,
)

resume_consultation_final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Tu es un assistant médical qui doit générer un résumé concis d'une consultation en utilisant les informations fournies ci-dessous. 
            Chaque résumé doit être personnalisé en utilisant le nom du patient et doit être bref mais complet. 
            Il doit inclure uniquement les éléments essentiels : le motif de consultation, les résultats de l'examen clinique, les prescriptions, les antécédents médicaux, les traitements en cours et tout autre point pertinent. 
            Le résumé doit être clair et sans ambiguïté, en veillant à ne pas omettre d'informations cruciales.

            Voici les informations reçues pour la consultation de [Nom du patient] :

            Anamnèse : [Description de l'anamnèse]
            Examen clinique : [Description de l'examen clinique]
            Examens paracliniques : [Résultats des examens paracliniques]
            Prescription : [Médicaments prescrits, examens recommandés, conseils, etc.]
            Antécédents médicaux : [Liste des antécédents médicaux]
            Traitement en cours : [Traitement actuel]
            Infos : [Âge, taille, poids, allergies].
         Résumé de la consultation :"

            Exemple d'utilisation :

            "Voici les informations reçues pour la consultation de Jean Dupont :

            Anamnèse : Jean consulte pour des douleurs abdominales et des ballonnements.
            Examen clinique : Abdomen sensible en fosse iliaque gauche, pas de masse palpable.
            Examens paracliniques : Aucun examen réalisé.
            Prescription : Antispasmodiques prescrits, régime hygiéno-diététique recommandé.
            Antécédents médicaux : Aucun antécédent notable.
            Traitement en cours : Aucun traitement en cours.
            Infos : Âge : 28 ans, Taille : 1m75, Poids : 70 kg, Allergies : Aucune.

            "voici un exemple de résumé":
                Résumé de la consultation : Jean Dupont, 28 ans, consulte pour des douleurs abdominales et des ballonnements. 
                    À l'examen clinique, un abdomen sensible en fosse iliaque gauche a été noté. Aucun examen paraclinique n'a été réalisé. 
                    Un antispasmodique a été prescrit, ainsi qu'un régime hygiéno-diététique.
         
            """
        ),
        few_shot_resume_consultation_prompt,
        ("human", "Voici la consultation : {input}")
    ]
)


prompt_format_paraclinique = ChatPromptTemplate.from_messages(
    [
        ("system", """ Tu es un expert en formatage de texte. tu dois absolument retourner un format JSON. C'est impératif.
                        A partir du texte qui te sera fourni tu devras énumerer tous les examens paracliniques effectués puis les formater en JSON
                        Voici le format de fichier attententu:
                        {{"paraclinique" : [
                            {{"nom": [Nom de l'examen],
                            "valeur" : [interpretation de l'examen],}}, ]
                        }}
         veuilles à toujours respecter le format de sortie c'est très important.
                    """),
        ("human", "Voici le texte à formater au format JSON : \n {input}")
    ]
)

prompt_format_clinique = ChatPromptTemplate.from_messages(
    [
        ("system", """ Tu es un expert en formatage de texte. tu dois absolument retourner un format JSON. C'est impératif.
                        A partir du texte qui te sera fourni tu devras énumerer tous les examens cliniques effectués puis les formater en JSON
                        Voici le format de fichier attententu:
                        {{"clinique" : [
                            {{"nom": [Nom de l'examen],
                            "valeur" : [valeur],}}, ]
                        }}
         veuilles à toujours respecter le format de sortie c'est très important.
                    """),
        ("human", "Voici le texte à formater au format JSON : \n {input}")
    ]
)

prompt_format_prescription = ChatPromptTemplate.from_messages(
    [
        ("system", """ Tu es un expert en formatage de texte. tu dois absolument retourner un format JSON. C'est impératif.
                        A partir du texte qui te sera fourni tu devras énumerer les traitements puis les formater en JSON
                        
                        Voici le format de fichier attententu:
         
                        {{
                            "traitement": {{ 
                                "laboratoire": [
                                    {{"nom": "Hémogramme complet", "observation": "Rechercher des signes d'inflammation"}}
                                ],
                                "imagerie": [
                                    {{"nom": "Radiographie des articulations", "observation": "Rechercher des signes de dégâts articulaires"}}
                                ],
                                "ophtalmologie": [],
                                "medicaments": [
                                    {{"nom": "Aspirine", "dose": "100mg", "posologie": "1 comprimé par jour", "duree": "7 jours"}}
                                ],
                                "recommendation": "vers qui le patient à été recommendé. laisser vide le cas écheant",
                                "autres": [{{"nom": "titre de l'observation", "observation": "detail de l'observation"}}]
                            }}
                    }}
                veuilles à toujours respecter le format de sortie c'est très important. Marque pas un champ vide les informations manquantes.
                    """),
        ("human", "Voici le texte à formater au format JSON : \n {input}")
    ]
)



# """
# Vous êtes un assistant spécialisé en analyse de conversations médicales. Votre tâche est d'extraire toutes les informations liées à l'examen clinique mentionnées dans une conversation. Concentrez-vous sur les éléments suivants :

#             Symptômes décrits par le patient (par exemple, douleur, fièvre, fatigue).
#             Signes physiques observés par le médecin (par exemple, rougeur, gonflement, anomalies visibles).
#             Actions réalisées par le médecin (par exemple, palpation, auscultation, inspection).
#             Résultats obtenus lors de l'examen clinique (par exemple, présence d'une masse, rythme cardiaque anormal).
#             Retournez les informations extraites sous la forme JSON c'est obligatoire. voici un exemple :

#         {{
#         "Examen clinique" :[

#             {{"Symptômes décrits" : [Liste des symptômes]
#             "Signes physiques" observés : [Liste des signes]
#             "Actions réalisées" : [Liste des actions]
#             "Résultats obtenus" : [Résumé des résultats]}},
#         ]
#         }}
#             Voici un exemple de conversation pour illustrer :

#             Conversation : Patient : J'ai ressenti une douleur intense au ventre depuis hier soir.
#             Médecin : D'accord, je vais palper votre abdomen pour voir s'il y a une masse ou une sensibilité.
#             Médecin : Je remarque une sensibilité accrue au niveau de l'abdomen inférieur droit.

#             Extraction des informations :

#             Symptômes décrits : Douleur intense au ventre.
#             Signes physiques observés : Sensibilité accrue au niveau de l'abdomen inférieur droit.
#             Actions réalisées : Palpation de l'abdomen.
#             Résultats obtenus : Pas de masse détectée, mais sensibilité confirmée.
#             Appliquez cette méthode à la conversation suivante et fournissez les informations sous le format spécifié.
# """