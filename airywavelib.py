# Airywaves python library
import numpy as np

def findWaveNumber(waveFrequency, waterDepth=None, IterMax=200, TOL=10.e-5):
    '''
    This function computes the wave number for a given wave frequency and water depth, by solving the dispersion relation for finite water depth.
    omega: [rad/s] Wave frequency
    h    : [m] Water depth
    '''
    omg = waveFrequency
    h = waterDepth
    g = 9.81
    k0 = omg** 2 / g
        
    if waterDepth is None:
        '''Deep water'''
        return k0

    else:
        '''Finite water depth'''
        for ii in range(1, IterMax):
            assert (ii < IterMax-1), "Solution did not converge."
            kh = k0*h
            F = g * k0 / omg**2 - 1.0 / np.tanh(kh)
            dF = g / omg**2 + 1.0 / np.sinh(kh) ** 2

            # Update estimate
            k = k0 - F / dF

            # Check residual and break if tolerance limit is reached
            epsilon = np.abs(k - k0) / k0;
            if epsilon < TOL:
                break
            else:
                k0 = k
        return k
		


class airywave:
    def __init__(self, wavePeriod, waveAmplitude, waterDepth=None):
        
        assert (wavePeriod >= 0),"Wave period must be positive!"
        assert (waveAmplitude >= 0),"Wave amplitude must be positive!"

        self.T = wavePeriod
        self.A = waveAmplitude
        self.h = waterDepth
        
        # Constants:
        self.g = 9.81
        self.rho = 1000.

        # Derived parameters:
        self.omega = 2*np.pi/self.T    # Wave circular frequency 
        self.k0 = self.omega**2/self.g # Deep water wave number
        self.l0 = 2*np.pi/self.k0      # Deep water wave length        
        
        if waterDepth is not None:
            # Water depth is given as input parameter...
            assert (waterDepth >= 0),"Water depth must be positive!"
        else:
            self.h = 10.*self.l0  # Set deep water as 10 times deep water wave length

        # Compute the wave number for finite water deth
        self.k = findWaveNumber(self.omega, self.h)
        self.wavelen = 2*np.pi/self.k

        self.steepness = 2*self.A/self.wavelen # Wave steepness H/lambda
        
    def getParticleVelocity(self,x,z,t):
        
        # Check if z-coordinate(s) is out of range. (should be in the range [-h,0])
        if isinstance(z, type(np.array)):
            assert (np.max(z) <= 0. and np.min(z) >= -self.h), "Input z-coordinate is out of range!"
        if isinstance(z, float):
            assert (z <= 0. and z >= -self.h), "Input z-coordinate is out of range!"
            
        fz1 = np.cosh(self.k*(z+self.h))/np.sinh(self.k*self.h)
        u = self.omega*self.A*fz1*np.cos(self.k*x-self.omega*t)
        
        fz2 = np.sinh(self.k*(z+self.h))/np.sinh(self.k*self.h)
        w = self.omega*self.A*fz2*np.sin(self.k*x-self.omega*t)
        
        return u, w
        
    
    def getParticleAcceleration(self,x,z,t):

        # Check if z-coordinate(s) is out of range. (should be in the range [-h,0])
        if isinstance(z, type(np.array)):
            assert (np.max(z) <= 0. and np.min(z) >= -self.h), "Input z-coordinate is out of range!"
        if isinstance(z, float):
            assert (z <= 0. and z >= -self.h), "Input z-coordinate is out of range!"
            
        fz1 = np.cosh(self.k*(z+self.h))/np.sinh(self.k*self.h)
        ax = self.omega**2*self.A*fz1*np.sin(self.k*x-self.omega*t)
        
        fz2 = np.sinh(self.k*(z+self.h))/np.sinh(self.k*self.h)
        az = -self.omega**2*self.A*fz2*np.cos(self.k*x-self.omega*t)
        
        return ax, az
    
    def getDynamicPressure(self,x,z,t):

        # Check if z-coordinate(s) is out of range. (should be in the range [-h,0])
        if isinstance(z, type(np.array)):
            assert (np.max(z) <= 0. and np.min(z) >= -self.h), "Input z-coordinate is out of range!"
        if isinstance(z, float):
            assert (z <= 0. and z >= -self.h), "Input z-coordinate is out of range!"
        
        fz = np.cosh(self.k*(z+self.h))/np.cosh(self.k*self.h)
        p = self.rho*self.g*self.A*fz*np.cos(self.k*x-self.omega*t)
        
        return p
        
    def getSurfaceElevation(self,x,t):
        
        zeta = self.A*np.cos(self.k*x-self.omega*t)
        return zeta
    
    def getParticleMotion(self,x,z,t):

    # Check if z-coordinate(s) is out of range. (should be in the range [-h,0])
        if isinstance(z, type(np.array)):
            assert (np.max(z) <= 0. and np.min(z) >= -self.h), "Input z-coordinate is out of range!"
        if isinstance(z, float):
            assert (z <= 0. and z >= -self.h), "Input z-coordinate is out of range!"
    
        xp = self.A*np.cosh(self.k*(z+self.h))/np.sinh(self.k*self.h)*(np.sin(self.k*x-self.omega*t) - np.sin(self.k*x))
        yp = self.A*np.sinh(self.k*(z+self.h))/np.sinh(self.k*self.h)*(np.cos(self.k*x-self.omega*t) - np.cos(self.k*x))
        
        return xp, yp