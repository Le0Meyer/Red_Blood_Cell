
# Modèle ne tenant pas compte de la solution tampon

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Ca serait logique que le pH intra varie beaucoup plus que le pH extra ( et pas symétrie), car le volume extra est beaucoup plus important ( 90% )
# Ci-dessus nouvelle forme de pHm qui redonne quelque chose qui varie peu.

def func(y,t,Ht0,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB,QtotH) :
    # Docstring de la fonction
    # Corps de la fonction
#	# Le vecteur y est égal à (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY)
#	QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY = y
	# Le vecteur y est égal à (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY)
	QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY = y
	
	# Equation des différents flux
	
	FluxPNa = -PhiMaxNa*(((QNa/Vw)/((QNa/Vw)+0.2*(1+(QK/(Vw*8.3)))))**3) * ((CmK/(CmK + 0.1 * (1 + (CmK/18))))**2)
	
	FluxPK  = -FluxPNa/1.5
	
	FluxLNa = -PLNa * ((QNa/Vw) - CmNa)
	
	FluxLK  = -PLK  * ((QK/Vw)  - CmK)

	# Equation des flux électro-diffusifs

	E = - R*T/F * np.log(( PGNa*QNa/Vw + PGK*QK/Vw + PGA*CmA ) / ( PGNa*CmNa + PGK*CmK + PGA*QA/Vw ))
	
	FluxGNa = -PGNa * FsurRT * E * (QNa/Vw - CmNa * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))
	
	FluxGK  = -PGK  * FsurRT * E * (QK/Vw  - CmK  * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))

	FluxGA  = +PGA  * FsurRT * E * (QA/Vw  - CmA  * np.exp(+ FsurRT * E))/(1 - np.exp(+ FsurRT * E))

	# Equation ...

	FluxCo  = -kCo  * (((QA/Vw)**2) * (QNa/Vw) * (QK/Vw) - d * (CmA**2) * CmNa * CmK)
	
	# Equation du flux HA

	#CmH     = KB * CmHB / ( CmB - CmHB )
	CmH = (-(QH*Ht0) + QtotH)/(1-Ht0)
	FluxHA  = -kHA  * (((QA * QH)/(Vw**2)) - CmA * CmH)
	
	# Variation des quantités Q
	
	dQNadt  = FluxPNa + FluxLNa + FluxGNa + FluxCo
#	dQNadt  = 0. # since the net Na flux is negligible (p. 70)
	
	dQKdt   = FluxPK  + FluxLK  + FluxGK  + FluxCo
	
	dQAdt   = FluxGA  + FluxHA  + FluxCo
	
	dQHdt   = FluxHA
	
	# Variation du volume intra-cellulaire
	# avec formule léna
	dVwdt   =  (dQKdt + dQAdt)/(CmNa + CmK + CmA + CmB + CmY)

	# Variation de Ht ?
	Ht = Ht0 * np.exp(Vw - Vw0)



	# Variation des concentrations extra-cellulaires Cm
	
	dCmNadt = (Ht/(1 - Ht)) * (dVwdt*CmNa - dQNadt)
	
	dCmKdt  = (Ht/(1 - Ht)) * (dVwdt*CmK  - dQKdt)
	
	dCmAdt  = (Ht/(1 - Ht)) * (dVwdt*CmA  - dQAdt)
	
	dCmHBdt = (Ht/(1 - Ht)) * (dVwdt*CmHB - dQHdt)  
	
	dCmBdt  = (Ht/(1 - Ht)) * (dVwdt*CmB) 	 #( *0.001 a une meilleure tête)

#	dCmHdt  =  KB * ((dCmHBdt * CmB - dCmBdt * CmHB)/((CmB-CmHB)**2))
	
	dCmYdt  = (Ht/(1 - Ht)) * (dVwdt*CmY)


#	# Vecteur final dydt issu de y = (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY)
#	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHdt,dCmHBdt,dCmBdt,dCmYdt]

	# Vecteur final dydt issu de y = (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY)
	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHBdt,dCmBdt,dCmYdt]

	return dydt

# On fixe les constantes
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

# Constante qui varie
PGA     = 2		# 1/h	# 0.2 a 200

# Constantes qui changent d'un cas à l'autre
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

# Condition initiale

# (		 QNa,	QK,			QA,			QH,						 Vw,  CmNa,CmK, CmA,
y0  = np.array([10 * Vw0, 140 * Vw0, 95 * Vw0, 1000 * 10**(-7.26) * Vw0, Vw0, 140., 5., 131.,
5.86, 10., 10.])
# CmHB, CmB, CmY)
# 8,    9,   10

# Vecteur de temps
tmin = 0.
tmax = 1.
t   = np.linspace(tmin, tmax, 1001)

plt.figure(figsize=(12, 9), dpi=80)


	# Appel à odeint
sol = odeint(func, y0, t, args=(Ht0,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB,QtotH))
	
plt.subplot(2, 3, 1)
	#plt.plot(t, sol[:,0], label='QNa')
plt.plot(t, sol[:,3], label='QH')
	#plt.plot(t, sol[:,4], label='Vw')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2, 3, 2)			## (QNa,QK,QA,QH,Vw, CmNa,CmK, CmA,CmHB, CmB, CmY)
plt.plot(t, sol[:,2], label='QA')	## ( 0, 1,  2, 3, 4, 5,     6 , 7,  8,   9,    10)
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(t, - np.log10(10**-3 * sol[:,3]/ sol[:,4] ), label='pHc') # pHc = - log10(QH/(1000*Vw))	
					## plt.plot(t, - np.log10(sol[:,8]/1000), '--', label='pHm')
					## plt.plot(t, - np.log10(10**-3 * KB * sol[:,8]/(sol[:,9] - sol[:,8]) ), '--', label='pHm')
plt.plot(t, - np.log10((QtotH - (sol[:,3]*0.1))*0.001/(0.9)), '--', label='pHm')	
plt.legend(loc='best')
plt.grid(True)
	

plt.subplot(2, 3, 4)
#plt.plot(t, sol[:,8], label='CmHB')
plt.plot(t, (QtotH - (sol[:,3]*0.1))*0.001/(1-0.1), label='int')

plt.legend(loc='best')
plt.grid(True)
					
plt.subplot(2, 3, 5)
plt.plot(t, 10**-3 * KB * sol[:,8] / ( sol[:,9] - sol[:,8] ), label='quot. pHm')  # quotient env = 3.98930026755e-08 (variation 10-14)

plt.legend(loc='best')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(t, 10**-3 * sol[:,3]/ sol[:,4], label='quot. pHc')	# quotient env = 8.46298649023e-08 ( variation 10-8)
#	plt.plot(t, Ht0 * np.exp(sol[:,4] - Vw0), label='Ht')
plt.legend(loc='best')
plt.grid(True)
	
plt.show()








#plt.plot(t, - R*T/F * 10**3 * np.log(( PGNa*sol[:,0]/sol[:,4] + PGK*sol[:,1]/sol[:,4] + PGA*sol[:,7] ) / ( PGNa*sol[:,5] + PGK*sol[:,6] + PGA*sol[:,2]/sol[:,4] )), label='E')
