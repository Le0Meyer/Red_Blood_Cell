# Modèle d'échanges ioniques et de régulation dans le globule rouge



# IMPORTATION des bibliothèques Python qui nosu serons utiles par la suite


from scipy.integrate import odeint # Ici on importe seulement la commande odeint (solveur de système d'edo) de la bibliothèque scipy.integrate, car c'est la seule qui nous intéresse.


import numpy as np # On importe la bibliothèque numpy, qui contient de nombreuses fonctions mathématiques, et on lui associe le raccourci np


import matplotlib.pyplot as plt # On importe la bibliothèque matplotlib.pyplot, qui va nous servvir à tracer nos courbes, et on lui associe le raccourci plt

#DEBUT DU PROGRAMME


# Pour fonctionner, le solveur odeint a besoin de plusieurs entrées : une fonction qui prend en entrée le vecteur y(t) de nos fonctions d'intérêt et qui ressort le vecteur dy(t)/dt dérivé de celui-ci, le vecteur y(0) des valeurs de nos fonctions au temps t=0, un intervalle de temps [0,T] découpé en p parties, et finalement les variables de notre système.

# Dans un code python, on commence généralement par écrire les fonctions qui nous serons utiles pour la suite. On commence donc par l'écriture de la fonction qu'odeint va utiliser pour résoudre le système. Comme vu précédemment, cette fonction prend le vecteur y(t) en entrée mais également l'intervalle de temps [0,T] découpé en p parties et les varaibles de notre système. Elle va ensuite resortir dy(t)/dt, le vecteur des fonctions dérivées, toutes exprimées grâce aux fonctions du vecteur y(t).


def func(y,t,Ht0,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB,QtotH) :# Ici on déclare que l'on écrit une fonction à l'aide de "def" puis on lui donne un nom et on précise ses arguments (= ce qu'elle prend en entrée).

	# On a les correspondances suivantes :
	# y = notre vecteur de fonctions
	# t = notre intervalle [0,T] découpé en p parties
	# Les autres entrées sont les variables de notre système.

	# On commence par lui indiquer les noms de nos fonctions pour pouvoir les réutiliser par la suite. Le vecteur y est ainsi égal à (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY)mY)
	
	QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY = y
	
	# On va ensuite expliciter toutes les formules de notre système une par une.
	
	# On commence par écrire toutes les équations qui nous serons utiles dans l'écriture du système.
	
	# Equation des flux de la pompe Na:K
	
	FluxPNa = -PhiMaxNa*(((QNa/Vw)/((QNa/Vw)+0.2*(1+(QK/(Vw*8.3)))))**3) * ((CmK/(CmK + 0.1 * (1 + (CmK/18))))**2)
	
	FluxPK  = -FluxPNa/1.5
	
	# Equation des flux de diffusion
	
	FluxLNa = -PLNa * ((QNa/Vw) - CmNa)
	
	FluxLK  = -PLK  * ((QK/Vw)  - CmK)

	# Equation des flux électro-diffusifs

	E = - R*T/F * np.log(( PGNa*QNa/Vw + PGK*QK/Vw + PGA*CmA ) / ( PGNa*CmNa + PGK*CmK + PGA*QA/Vw ))
	
	FluxGNa = -PGNa * FsurRT * E * (QNa/Vw - CmNa * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))
	
	FluxGK  = -PGK  * FsurRT * E * (QK/Vw  - CmK  * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))

	FluxGA  = +PGA  * FsurRT * E * (QA/Vw  - CmA  * np.exp(+ FsurRT * E))/(1 - np.exp(+ FsurRT * E))

	# Equation des flux du cotransporteur Na:K:2A

	FluxCo  = -kCo  * (((QA/Vw)**2) * (QNa/Vw) * (QK/Vw) - d * (CmA**2) * CmNa * CmK)
	
	# Equation du flux HA

	CmH = (-(QH*Ht0) + QtotH)/(1-Ht0)
	
	FluxHA  = -kHA  * (((QA * QH)/(Vw**2)) - CmA * CmH)
	
	# Après avoir écrit les équations des flux, on va pouvoir écrire les différentes équations différentielles ordinaires de notre système pour ensuite les replacer dans le vecteur final dydt
	
	# Variation des quantités Q
	
	dQNadt  = FluxPNa + FluxLNa + FluxGNa + FluxCo
	
	# Dans certains cas, il peut être utile de considérer que le flux de Na est négligeable (cf p.70 de la publication), donc égal à 0. Pour cela, on commentera la ligne précédente et on décommentera la ligne suivante :
	#dQNadt  = 0.
	
	dQKdt   = FluxPK  + FluxLK  + FluxGK  + FluxCo
	
	dQAdt   = FluxGA  + FluxHA  + FluxCo
	
	dQHdt   = FluxHA
	
	# Variation du volume intra-cellulaire
	
	dVwdt   =  (dQKdt + dQAdt)/(CmNa + CmK + CmA + CmB + CmY)

	# Variation de Ht 
	
	Ht = Ht0 * np.exp(Vw - Vw0)

	# Variation des concentrations extra-cellulaires Cm
	
	dCmNadt = (Ht/(1 - Ht)) * (dVwdt*CmNa - dQNadt)
	
	dCmKdt  = (Ht/(1 - Ht)) * (dVwdt*CmK  - dQKdt)
	
	dCmAdt  = (Ht/(1 - Ht)) * (dVwdt*CmA  - dQAdt)
	
	dCmHBdt = (Ht/(1 - Ht)) * (dVwdt*CmHB - dQHdt)  
	
	dCmBdt  = (Ht/(1 - Ht)) * (dVwdt*CmB)
	
	dCmYdt  = (Ht/(1 - Ht)) * (dVwdt*CmY)

	# Maintenant que on a explicité toutes les dérivées de nos fonctions d'intérêts, on peut les rassembler dans le vecteur dydt qui sera la sortie de notre fonction

	# Vecteur final dydt issu de y = (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY)
	
	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHBdt,dCmBdt,dCmYdt]

	return dydt

# Notre fonction en elle-même ne sert à rien, en effet pour l'utiliser il va falloir faire appel à elle, avec différents paramètres que nous allons donc écrire maintenant :

# On écrit les paramètres :

Ht0     = 0.1		# 1
PhiMaxNa= 8.99		# mmol/l*h
F       = 96485
E       = -0.0086	# V
R       = 8.314
T       = 310		# K
FsurRT  = F / (R * T) # 1/V
d       = 1.05		# 1
fHb     = 2.78		# 1
QHb     = 5			# mmol/l
QMg     = 2.5		# mmol/l
QX      = 19.2		# mmol/l
KB      = 10**-4.55	# mmol/l
Vw0     = 0.7		# 1
QtotH   = (4 * (10**-5) * 0.9) + (1000 * 10**(-7.26) * Vw0 * 0.1) 
PGA     = 2		# 1/h	# 0.2 a 200

# Il ya quelques constantes qui changent en fonction des cas et qui ont une influence sur la valeurs de certaines variables. Pour passer d'un cas à l'autre il suffit de commenter les lignes inutiles et de décommenter les lignes qui nous intéressent !

# Cas 1 : avec fG = 0.1 et mode 'off'
#PLNa    = 0.0180	# 1/h
#PLK     = 0.0116	# 1/h
#PGNa    = 0.0017	# 1/h
#PGK     = 0.0015	# 1/h
#kCo     = 10**-9	# 1
#kHA     = 1		# 1

# Cas 2 : avec fG = 0.9 et mode 'on'
PLNa    = 0.0020	# 1/h
PLK     = 0.0013	# 1/h
PGNa    = 0.0151	# 1/h
PGK     = 0.0138*10**4	# 1/h
kCo     = 10**-6	# 1
kHA     = 10**9		# 1

# Pour résoudre notre système il nous faut un vecteur y(0) comportant les conditions initiales de chaque fonctions.
# (		QNa     , QK       , QA      , QH                      , Vw , CmNa,CmK, CmA ,
y0  = np.array([10 * Vw0, 140 * Vw0, 95 * Vw0, 1000 * 10**(-7.26) * Vw0, Vw0, 140., 5., 131.,
5.86, 10., 10.])
#CmHB,CmB, CmY)

# On indique ensuite notre intervalle d'étude. Pour cela, on fixe un tmin (généralement = 0) qui fixe le temps pour nos conditions initiales, puis un tmax qui ferme notre intervalle.
# Vecteur de temps
tmin = 0.
tmax = 1.

# La commande suivante génère un intervalle comprenant 1001 valeurs réparties linéairement entre tmin et tmax. Pour une valeur tmax - tmin grande, il est judicieux d'augmenter le nombre de valeurs.

t   = np.linspace(tmin, tmax, 1001)

# On va maintenant faire appel au solveur pour résoudre notre système.
# Il prend donc en entrée notre fonction écrite en début de programme, le vecteur des conditions intiales, l'intervalle d'étude, et les variables du système.

sol = odeint(func, y0, t, args=(Ht0,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB,QtotH))

# la variable sol sera donc une matrice où mes colones représentent les valeurs de nos fonctions pour chaque valeurs de l'intervalle d'étude.
	
# On va ensuite tracer les différentes courbes.
plt.figure(figsize=(12, 9), dpi=80)

# On défini des subplot, qui sont subdivision de notre fenètre d'afichage des courbes. Ici on a donc découper la fenêtre en 2x3 cases et on trace dans la case 1 :
plt.subplot(2, 3, 1)
#On trace notre fonction en indiquant les valeurs en abscisses puis en ordonnées et une légende
plt.plot(t, sol[:,2])
# On nomme nos axes
plt.xlabel('Temps en heure')
plt.ylabel('QA en mmol/loc')
#On active la grille
plt.grid(True)

plt.subplot(2, 3, 2)		
plt.plot(t, sol[:,1])
plt.xlabel('Temps en heure')
plt.ylabel('QK en mmol/loc')
plt.grid(True)

plt.subplot(2, 3, 3)	
plt.plot(t, - np.log10(10**-3 * sol[:,3]/ sol[:,4] ), label='pHc') 
plt.plot(t, - np.log10((QtotH - (sol[:,3]*0.1))*0.001/(0.9)), '--', label='pHm')
plt.ylim(6.2, 7.5)
plt.legend(loc='best')	
plt.xlabel('Temps en heure')
plt.ylabel('pH (sans unité)')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(t, - R*T/F * 10**3 * np.log(( PGNa*sol[:,0]/sol[:,4] + PGK*sol[:,1]/sol[:,4] + PGA*sol[:,7] ) /
            ( PGNa*sol[:,5] + PGK*sol[:,6] + PGA*sol[:,2]/sol[:,4] )))
plt.xlabel('Temps en heure')
plt.ylabel('E en mV')
plt.grid(True)
	
plt.subplot(2, 3, 5)
plt.plot(t, sol[:,4]/Vw0)
plt.xlabel('Temps en heure')
plt.ylabel('Vw/Vw0 (sans unité)')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(t, sol[:,3]*10**3/Vw0)
plt.ylim(0.05, 0.14)
plt.xlabel('Temps en heure')
plt.ylabel('QH en 10^(-3).mmol/loc')
plt.grid(True)

#On affiche notre fenêtre.
plt.show()
