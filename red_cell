#premier essai pour le stage

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def func(y,t,Ht,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB) :

	# Le vecteur y est égal à (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY)
	
	
#	#QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY = y
	#QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY = y
	
	#QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY = y
	QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHB,CmB,CmY,Q_ = y
	
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

	CmH     = KB * CmHB / ( CmB - CmHB )
	
	FluxHA  = -kHA  * (((QA * QH)/(Vw**2))- CmA * CmH)
	
	# Variation des quantités Q
	
	dQNadt  = FluxPNa + FluxLNa + FluxGNa + FluxCo
	
	dQKdt   = FluxPK  + FluxLK  + FluxGK  + FluxCo
	
	dQAdt   = FluxGA  + FluxHA  + FluxCo
	
	dQHdt   = FluxHA
	
	dQ_dt   = dQHdt
	
	
	
	# Variation du volume intra-cellulaire
	
	#dVwdt   = ((CmNa + CmK + CmA + CmB + CmY)*(dQNadt + dQKdt + dQAdt) - (fHb * QHb + QNa + QK + QA + QMg + QX) * (dCmNadt + dCmKdt + dCmAdt + dCmBdt + dCmYdt))/((CmNa + CmK + CmA + CmB + CmY)**2)
	#avec formule léna
	#dVwdt   =  ((dQNadt + dQKdt + dQAdt)/(CmNa + CmK + CmA + CmB + CmY))
	#MODIFICATION
	
	dVwdt   =  ((dQKdt + dQAdt)/(CmNa + CmK + CmA + CmB + CmY))
	
	# Variation des concentrations extra-cellulaires Cm
	
	dCmNadt = (Ht/(1 - Ht)) * (dVwdt*CmNa - dQNadt)
	
	dCmKdt  = (Ht/(1 - Ht)) * (dVwdt*CmK  - dQKdt)
	
	dCmAdt  = (Ht/(1 - Ht)) * (dVwdt*CmA  - dQAdt)
	
	dCmHBdt = (Ht/(1 - Ht)) * (dVwdt*CmHB - dQHdt)
	
	dCmBdt  = (Ht/(1 - Ht)) * (dVwdt*CmB)
	
#	dCmHdt  =  KB * ((dCmHBdt * CmB - dCmBdt * CmHB)/((CmB-CmHB)**2))
	
	dCmYdt  = (Ht/(1 - Ht)) * (dVwdt*CmY)
	
	# Vecteur final dydt issu de y = (QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmH,CmHB,CmB,CmY)
#	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHdt,dCmHBdt,dCmBdt,dCmYdt]
	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHBdt,dCmBdt,dCmYdt,dQ_dt]		

	return dydt

# Valeurs initiales

# On fixe les constantes
Ht      = 0.1	# 1
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

# Constante qui varie
PGA     = 0.2		# 1/h	# 0.2 a 200

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
PGK     = 0.0138	# 1/h
kCo     = 10**-6	# 1
kHA     = 10**9		# 1

# Vecteur initial
## (				QNa,	QK,			QA,			QH,						 Vw,  CmNa,CmK, CmA,
#y0  = np.array([10 * Vw0, 140 * Vw0, 95 * Vw0, 1000 * 10**(-7.26) * Vw0, Vw0, 140., 5., 131.,
#1000 * 10**(-7.4), 5.86, 10., 10.])
## CmH,			   CmHB, CmB, CmY)
# (				QNa,	QK,			QA,			QH,						 Vw,  CmNa,CmK, CmA,
#y0  = np.array([10 * Vw0, 140 * Vw0, 95 * Vw0, 1000 * 10**(-7.26) * Vw0, Vw0, 140., 5., 130.9,
#5.86, 10., 10.])
# CmHB, CmB, CmY)

## CmH,			   CmHB, CmB, CmY)
# (				QNa,	QK,			QA,			QH,						 Vw,  CmNa,CmK, CmA,
y0  = np.array([10 * Vw0, 140 * Vw0, 95 * Vw0,  1000 * 10**(-7.26) * Vw0, Vw0, 140., 5., 5.,
5.86, 10., 135.9, 38.5])
# CmHB, CmB, CmY, Q_)


t   = np.linspace(0.,1.,1001)

sol = odeint(func, y0, t, args=(Ht,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,fHb,QHb,QMg,QX,KB))


E = - R*T/F * np.log(( PGNa*sol[:,0]/sol[:,4] + PGK*sol[:,1]/sol[:,4] + PGA*sol[:,7] ) / ( PGNa*sol[:,5] + PGK*sol[:,6] + PGA*sol[:,2]/sol[:,4] ))

plt.figure()
plt.subplot(2, 2, 1)
#plt.plot(t, sol[:,0] , 'green', label='QNa')
#plt.plot(t, sol[:,1] , 'black', label='QK')
plt.plot(t, sol[:,2] , label='QA')
#plt.plot(t, sol[:,4] , label='Vw')
#plt.plot(t, sol[:,8] , label='CmHB')
#plt.plot(t, sol[:,9] , label='CmB')
#plt.plot(t, sol[:,11] , label='Q_')
plt.plot(t,E,label='E')
plt.legend(loc='best')
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(t, - np.log10(sol[:,3]/(1000*sol[:,4])), label='pHc') # pHc = - log10(QH/(1000*Vw))
plt.plot(t, 6.85 + (sol[:,11] + 18) / (-10 * 5), label='pHcT')
plt.plot(t, - np.log10(KB *0.001* sol[:,8] / ( sol[:,9] - sol[:,8] )), label='pHm')
plt.legend(loc='best')
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(t, sol[:,4]/Vw0, label ='V/V0')
plt.legend(loc='best')
plt.grid()
plt.subplot(2,2,4)
plt.plot(t,1 + 0.0645 * (5/sol[:,4]) + 0.026 * (5/sol[:,4])**2, label='fHb')
plt.legend(loc='best')
plt.grid()
plt.show()
