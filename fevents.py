import datetime
from pylab import *
import observer
import ephem
import pandas as pd
import matplotlib.dates as mdates
import os

def get_moon_ill(obs,time):
    obs2     = ephem.Observer()
    obs2.lat = obs.site_info.latitude
    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude
    obs2.elevation = obs.site_info.elevation
    m = ephem.Moon()
    obs2.date = time
    m.compute(obs2)

    pfm = ephem.previous_full_moon(obs2.date)  
    nfm = ephem.next_full_moon(obs2.date)  
    pnm = ephem.previous_new_moon(obs2.date)
    nnm = ephem.next_new_moon(obs2.date)
    if obs2.date - pnm < obs2.date - pfm:
        moon_state = 'crescent'
        lunation   = (obs2.date-pnm)/(nfm-pnm)
    else:
        moon_state = 'waning'
        lunation   = 1. - (obs2.date-pfm)/(nnm-pfm)

    return moon_state, lunation


def get_moon_elev(obs,times):
    obs2     = ephem.Observer()
    obs2.lat = obs.site_info.latitude

    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude
    obs2.elevation = obs.site_info.elevation
    altitudes = []
    for time in times:
        m = ephem.Moon()
        ttime = time #- datetime.timedelta(hours=4) 
        obs2.date = ttime
        m.compute(obs2)
        altitudes.append(m.alt*180./np.pi)
    return np.array(altitudes)

def get_moon_sep(time1,time2,ra,dec):
    obs = observer.Observer('saao')
    obs2 = ephem.Observer()
    obs2.lat = obs.site_info.latitude
    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude

    date = time1
    obs2.date = str(date)
    m = ephem.Moon(obs2)

    rasep    = np.sqrt((ra - m.ra*180./np.pi)**2)
    decsep   = np.sqrt((dec - m.dec*180/np.pi)**2)
    if rasep > 180:
        rasep = 360 - rasep
    if decsep > 180:
        decsep = 360 - decsep
    moonsep = np.sqrt( (rasep)**2 + (decsep)**2 )

    return moonsep

def iau_cal2jd(IY,IM,ID):
    IYMIN = -4799.
    MTAB = np.array([ 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
    J = 0
    if IY < IYMIN:
        J = -1
    else:
        if IM>=1 and IM <= 12:
            if IY%4 == 0:
                MTAB[1] = 29.
            else:
                MTAB[1] = 28.

            if IY%100 == 0 and IY%400!=0:
                MTAB[1] = 28.
            if ID < 1 or ID > MTAB[IM-1]:
                J = -3
            a = ( 14 - IM ) / 12
            y = IY + 4800 - a
            m = IM + 12*a -3
            DJM0 = 2400000.5
            DJM = ID + (153*m + 2)/5 + 365*y + y/4 - y/100 + y/400 - 32045 - 2400001.
        else:
            J = -2
    return DJM0, DJM, J

def get_mjd(date):
    year = date.year
    month = date.month
    day = date.day
    hour,minu,sec = date.hour,date.minute,date.second
    epoch       = 2000.
    mjd0,mjd,i = iau_cal2jd(int(year),int(month),int(day))
    ho = int(hour)
    mi = int(minu)
    se = float(sec)
    ut = float(ho) + float(mi)/60.0 + float(se)/3600.0
    mjd_start = mjd + ut/24.0
    return mjd_start

import datetime
from pylab import *
import observer
import ephem
import pandas as pd
import matplotlib.dates as mdates
import os

def get_moon_ill(obs,time):
    obs2     = ephem.Observer()
    obs2.lat = obs.site_info.latitude
    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude
    obs2.elevation = obs.site_info.elevation
    m = ephem.Moon()
    obs2.date = time
    m.compute(obs2)

    pfm = ephem.previous_full_moon(obs2.date)  
    nfm = ephem.next_full_moon(obs2.date)  
    pnm = ephem.previous_new_moon(obs2.date)
    nnm = ephem.next_new_moon(obs2.date)
    if obs2.date - pnm < obs2.date - pfm:
        moon_state = 'crescent'
        lunation   = (obs2.date-pnm)/(nfm-pnm)
    else:
        moon_state = 'waning'
        lunation   = 1. - (obs2.date-pfm)/(nnm-pfm)

    return moon_state, lunation


def get_moon_elev(obs,times):
    obs2     = ephem.Observer()
    obs2.lat = obs.site_info.latitude

    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude
    obs2.elevation = obs.site_info.elevation
    altitudes = []
    for time in times:
        m = ephem.Moon()
        ttime = time #- datetime.timedelta(hours=4) 
        obs2.date = ttime
        m.compute(obs2)
        altitudes.append(m.alt*180./np.pi)
    return np.array(altitudes)

def get_moon_sep(time1,time2,ra,dec):
    obs = observer.Observer('saao')
    obs2 = ephem.Observer()
    obs2.lat = obs.site_info.latitude
    if obs.site_info.longitude[0] == '-':
        obs2.lon = obs.site_info.longitude[1:]
    else:
        obs2.lon = '-'+obs.site_info.longitude

    date = time1
    obs2.date = str(date)
    m = ephem.Moon(obs2)

    rasep    = np.sqrt((ra - m.ra*180./np.pi)**2)
    decsep   = np.sqrt((dec - m.dec*180/np.pi)**2)
    if rasep > 180:
        rasep = 360 - rasep
    if decsep > 180:
        decsep = 360 - decsep
    moonsep = np.sqrt( (rasep)**2 + (decsep)**2 )

    return moonsep

def iau_cal2jd(IY,IM,ID):
    IYMIN = -4799.
    MTAB = np.array([ 31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])
    J = 0
    if IY < IYMIN:
        J = -1
    else:
        if IM>=1 and IM <= 12:
            if IY%4 == 0:
                MTAB[1] = 29.
            else:
                MTAB[1] = 28.

            if IY%100 == 0 and IY%400!=0:
                MTAB[1] = 28.
            if ID < 1 or ID > MTAB[IM-1]:
                J = -3
            a = ( 14 - IM ) / 12
            y = IY + 4800 - a
            m = IM + 12*a -3
            DJM0 = 2400000.5
            DJM = ID + (153*m + 2)/5 + 365*y + y/4 - y/100 + y/400 - 32045 - 2400001.
        else:
            J = -2
    return DJM0, DJM, J

def get_mjd(date):
    year = date.year
    month = date.month
    day = date.day
    hour,minu,sec = date.hour,date.minute,date.second
    epoch       = 2000.
    mjd0,mjd,i = iau_cal2jd(int(year),int(month),int(day))
    ho = int(hour)
    mi = int(minu)
    se = float(sec)
    ut = float(ho) + float(mi)/60.0 + float(se)/3600.0
    mjd_start = mjd + ut/24.0
    return mjd_start

def make_plot2(refdate, ra, dec, t0, per, dur, obname, dest='./'):

	sy,sm,sd = str(refdate.year),str(refdate.month),str(refdate.day)
	date = sy+'-'+sm+'-'+sd

	if refdate.month < 10:
		sm = '0'+sm
	if refdate.day < 10:
		sd = '0'+sd

	fig, ax = subplots(1,figsize=(20,6))
	fig.autofmt_xdate()

	obs = observer.Observer('saao')
	obs.almanac(date.replace('-','/'))
	tw1 = obs.almanac_data.evening_twilight_12().datetime()#.strftime('%Y-%m-%d %H:%M:%S')
	tw2 = obs.almanac_data.evening_twilight_18().datetime()#.strftime('%Y-%m-%d %H:%M:%S')
	tw3 = obs.almanac_data.morning_twilight_18().datetime()#.strftime('%Y-%m-%d %H:%M:%S')
	tw4 = obs.almanac_data.morning_twilight_12().datetime()#.strftime('%Y-%m-%d %H:%M:%S')
	ts1 = obs.almanac_data.sunset().datetime()#.strftime('%Y-%m-%d %H:%M:%S')
	ts2 = obs.almanac_data.sunrise().datetime()#.strftime('%Y-%m-%d %H:%M:%S')

	x1 = pd.to_datetime(np.array([ts1, tw1]))
	x2 = pd.to_datetime(np.array([tw1, tw2]))
	x3 = pd.to_datetime(np.array([tw2, tw3]))
	x4 = pd.to_datetime(np.array([tw3, tw4]))
	x5 = pd.to_datetime(np.array([tw4, ts2]))

	y1 = np.array([0,0])
	y2 = np.array([90,90])

	pos = 95

	if True:
		obs = observer.Observer('saao')
		obs.almanac(date.replace('-','/'))

		nra,ndec = coordsDeg_to_coords(ra,dec)
		target   = obs.target(obname, nra, ndec)
		obs.airmass(target)
		AM = np.array(obs.airmass_data[0].airmass)
		ALT = 0.5*np.pi - np.arccos(1./AM)
		ALT = 180.*ALT/np.pi
		utc = obs.airmass_data[0].utc
		reft = []
		pdt = []
		for t in utc:
			reft.append(get_mjd(t.datetime()))
			pdt.append(t.datetime())

		reft,pdt = np.array(reft),pd.to_datetime(np.array(pdt))
		II = np.where(ALT>10)[0]
		plot(pdt[II],ALT[II],'r')

		if True:
			t0 = t0 - 2400000.5
			phase = (reft-t0)/per
			phase = phase - phase.astype('int')

			IT1 = np.where( (phase > 1 - 0.5*dur/per)) [0]
			IT2 = np.where( (phase < 0.5*dur/per))[0]
			IT = np.hstack((IT1,IT2))
			if len(IT)>0:
				plot(pdt[IT],ALT[IT],'r',linewidth='4.')

			IT = np.where((phase > 0.5 - 0.5*dur/per) & (phase < 0.5 + 0.5*dur/per))[0]
			if len(IT)>0:
				plot(pdt[IT],ALT[IT],'g',linewidth='4.')


		moonsep = get_moon_sep(pdt[II[0]], pdt[II[-1]],ra, dec)
		text(pdt[II][0], pos, 'Moon: '+str(np.around(moonsep))+' deg', color="black")

	rng = pd.date_range(ts1, ts2, freq='15min')
	altitudes = get_moon_elev(obs,rng)
	I = np.where(altitudes>0)[0]
	if len(I)>0:
		plot(rng[I],altitudes[I],'y-')

	moon_state, moon_ill = get_moon_ill(obs,rng[int(0.5*len(rng))])
	text(ts1, 100, moon_state + ' moon, ' + str(int(np.around(moon_ill*100)))+' %')

	fill_between(x1, y1, y2, where=y2 >= y1, facecolor='lightblue', interpolate=True)
	fill_between(x2, y1, y2, where=y2 >= y1, facecolor='blue', interpolate=True)
	fill_between(x3, y1, y2, where=y2 >= y1, facecolor='darkblue', interpolate=True)
	fill_between(x4, y1, y2, where=y2 >= y1, facecolor='blue', interpolate=True)
	fill_between(x5, y1, y2, where=y2 >= y1, facecolor='lightblue', interpolate=True)
	axhline(30., color='k',linestyle='--')

	xlim(x1[0],x5[-1])
	ylim(0,90)

	ylabel('Altitude')

	xfmt = mdates.DateFormatter('%d-%m %H:%M')
	ax.xaxis.set_major_formatter(xfmt)

	fname = dest+'/futuretransit_'+str(refdate).split()[0]+'_'+obname+'.png'
	savefig(fname,bbox_inches='tight',format='png')
	return fname.split('/')[-1]

def coordsDeg_to_coords(ra,dec):
	ra = ra * 24./360.
	HH = int(ra)
	sHH = str(HH)
	if HH < 10:
		sHH = '0'+sHH
	mins = (ra - HH)*60.
	MM = int(mins)
	sMM = str(MM)
	if MM < 10:
		sMM = '0'+sMM
	segs = (mins - MM)*60.
	sSS = str(np.around(segs,3))
	if np.around(segs,3) < 10:
		sSS = '0'+sSS

	nra = sHH + ' ' + sMM + ' ' + sSS

	if dec < 0:
		sign = '-'
	else:
		sign = ''

	dec = np.absolute(dec)
	DD = int(dec)
	sDD = str(DD)
	if DD < 10:
		sDD = '0'+sDD
	mins = (dec - DD)*60.
	MM  = int(mins)
	sMM = str(MM)
	if MM < 10:
		sMM = '0'+sMM
	segs = (mins - MM)*60.
	sSS = str(np.around(segs,3))
	if np.around(segs,3) < 10:
		sSS = '0'+sSS
	ndec = sDD + ' ' + sMM + ' ' + sSS

	ndec = sign + ndec
	return nra,ndec

def is_visible(target,obs,date,maxAM=2.0,mintime=0.5,tel='saao'):
	obs = observer.Observer(tel)
	obs.almanac(date.replace('-','/'))
	obs.airmass(target)
	AM = np.array(obs.airmass_data[0].airmass)
	utc = np.array(obs.airmass_data[0].utc)
	I =  np.where(AM<maxAM)[0]

	if len(I)>0:
		tw1 = obs.almanac_data.evening_twilight_12().datetime().strftime('%Y-%m-%d %H:%M:%S')
		tw2 = obs.almanac_data.morning_twilight_12().datetime().strftime('%Y-%m-%d %H:%M:%S')
		ini = ephem.julian_date(utc[I][0])
		fin = ephem.julian_date(utc[I][-1])

		if ini < ephem.julian_date(tw1):
			ini = ephem.julian_date(tw1)
		if fin > ephem.julian_date(tw2):
			fin = ephem.julian_date(tw2)

		if fin - ini > mintime/24.:
			suc = True
		else:
			suc = False
		#print ini, fin, suc
		return suc
	else:
		return False

def get_time_up(target,obs,date,maxAM=2.0,tel='saao'):
	obs = observer.Observer(tel)
	obs.almanac(date.replace('-','/'))
	obs.airmass(target)
	AM  = np.array(obs.airmass_data[0].airmass)
	utc = np.array(obs.airmass_data[0].utc)
	I   =  np.where(AM<maxAM)[0]

	vecs = []
	vec = []
	for i in range(len(I)-1):
		if I[i]+1 == I[i+1]:
			vec.append(I[i])
		else:
			vec.append(I[i])
			vecs.append(vec)
			vec = []
	if len(vec) > 0:
		vec.append(I[-1])
	else:
		vec = [I[-1]]
	vecs.append(vec)

	mmax=0
	for vec in vecs:
		if len(vec) > mmax:
			I = np.array(vec)
			mmax = len(I)

	
	best = np.argmin(AM)
	ini,imid,fin = I[0],I[int(0.5*len(I))],I[-1]
	#print ephem.julian_date(obs.airmass_data[0].utc[ini]), obs.airmass_data[0].utc[fin]
	#print fds

	tw1 = obs.almanac_data.evening_twilight_12().datetime().strftime('%Y-%m-%d %H:%M:%S')
	tw2 = obs.almanac_data.morning_twilight_12().datetime().strftime('%Y-%m-%d %H:%M:%S')
	ini = ephem.julian_date(obs.airmass_data[0].utc[ini])
	fin = ephem.julian_date(obs.airmass_data[0].utc[fin])

	if ephem.julian_date(tw1) > ini:
		ini = ephem.julian_date(tw1)
	if ephem.julian_date(tw2) < fin:
		fin = ephem.julian_date(tw2)

	return ini,fin

def GetTransitsInNight(ra,dec,P,t0,dur,date='today',tel='saao',maxAM=2.0,obname='test',out=10.):
	if date == 'today':
		refdate = datetime.datetime.utcnow()
		if refdate.hour < 11:
			refdate -= datetime.timedelta(days=1) 
		sy,sm,ss = str(refdate.year),str(refdate.month),str(refdate.day)
		date = sy+'-'+sm+'-'+ss
	else:
		cos = date.split('-')
		refdate = datetime.datetime(int(cos[0]),int(cos[1]),int(cos[2]),13,0,0)
	obs = observer.Observer(tel)
	obs.almanac(date.replace('-','/'))
	nra,ndec = coordsDeg_to_coords(ra,dec)
	target   = obs.target(obname, nra, ndec)
	vis = is_visible(target,obs,date,maxAM=maxAM,tel=tel)
	if vis:
		jd1,jd2  = get_time_up(target,obs,date,maxAM=maxAM,tel=tel)
		if jd2 > jd1:
			intransit, transits = get_transits(P,t0,dur,jd1,jd2,delt=out)
			if intransit:
				return transits
			else:
				return []
	return []

def get_transits(P,t0,dur,jd1,jd2,delt=45.):
	timeet = np.arange(jd1,jd2,0.001)
	phaset = (timeet - t0)/P
	phaset = phaset - phaset.astype('int')
	I = np.where(phaset < 0.5)[0]
	phaset[I] = phaset[I] + 1.
	tphase = np.append(phaset[1:], phaset[0])
	J = np.where(tphase < phaset)[0]
	cond = True
	start = 0
	end = J[0] + 1
	count = 0
	intransit = 0
	result = []
	while cond:
		suc = False
		phase = phaset[start:end]
		timee = timeet[start:end]
		ingress1 = 1. - 0.5*dur/P - delt/(60.*24*P)
		ingress2 = 1. - 0.5*dur/P + 0.006
		egress1  = 1. + 0.5*dur/P - 0.006
		egress2  = 1. + 0.5*dur/P + delt/(60.*24*P)
		I1 = np.where(phase < ingress1)[0]
		I2 = np.where(phase > ingress2)[0]
		I3 = np.where(phase < egress1)[0]
		I4 = np.where(phase > egress2)[0]
		I5 = np.where((phase > ingress1) & (phase < egress2))[0]
		if len(I1) >0 and len(I2) >0:
			T1 = 1
		else:
			T1 = 0

		if len(I3) >0 and len(I4) >0:
			T2 = 1
		else:
			T2 = 0

		if len(I5)>1 and (T1>0 or T2>0):
			intransit += 1
			if len(result) == 0:
				result = np.array([T1,T2,timee[I5[0]],timee[I5[-1]],phase[I5[0]],phase[I5[-1]]])
			else:
				result = np.vstack((result, np.array([T1,T2,timee[I5[0]],timee[I5[-1]],phase[I5[0]],phase[I5[-1]]])))

		if end == len(phaset):
			cond = False
		elif end - 1 == J[-1]:
			start = end
			end = len(phaset)
			cond = True
		else:
			start = end
			end = J[count+1]
			cond = True
		count +=1
	return intransit, result


#ra  = 166.42973
#dec = -5.07942388889
#t0  = 2456649.54972
#dur = 0.22346
#per = 9.289715
#obname = 'WASP-106'
"""
ra  = 73.7665
dec = 18.6545361111
t0  = 2457825.35297
dur = 0.1675
per = 11.16724739
obname = 'K2-232'

ra  = 233.074341667
dec = -22.3582611111
t0  = 2458001.72138
dur = 0.152107835705
per = 14.893291
obname = 'CL001-15'

ra  = 192.1898
dec = -47.6136916667
t0  = 2457139.1672
dur = 0.2011
per = 16.254611
obname = 'HATS-17'

ra  = 163.032416667
dec = 0.493352777778
t0  = 2457906.84038
dur =  0.214830744352
per = 11.6336390737
obname = 'EPIC201498078'

ra  = 18.29291666666667
dec = -17.657777777777778
t0  = 2457701.3816
dur =  0.1774
per = 9.62468
obname = 'WASP-162'


ra  = 154.671083333
dec = 10.129027777777
t0  = 2457913.8049
dur =  0.2362 
per = 11.81433
obname = 'K2-234'

ra  = 203.105933333
dec = -42.4753
t0  = 2456921.14307
dur = 0.155
per = 11.55098
obname = 'WASP-130'
"""
ra  =  4.44641
dec = -66.35898
t0  = 2458332.0828
dur = 0.3566
per = 11.53508
obname = 'HD1397'

os.system('mkdir out/')
os.system('mkdir out/'+obname)
all_transits = []
base = datetime.datetime.utcnow()#.strftime('%Y-%m-%d %H:%M:%S')
arr = np.array([base + datetime.timedelta(days=i) for i in xrange(1000)])
good_dates = []
plots = []
for datet in arr:
	date = datet.strftime('%Y-%m-%d')
	transits = GetTransitsInNight(ra,dec,per,t0,dur,obname=obname,date=date,out=10.)
	#print transits
	if len(transits)>0:
		good_dates.append(date)
		if len(transits.shape) == 1:
			transits = np.hstack((transits,date))
			if len(all_transits) == 0:
				all_transits = np.array(transits)
			else:
				all_transits = np.vstack([all_transits,transits])
			fname = make_plot2(datet, ra, dec, t0, per, dur, obname,dest='out/'+obname+'/')
			plots.append(fname)
				   
		else:
			for transit in transits:
				transit = np.hstack((transit,date))
				if len(all_transits) == 0:
					all_transits = np.array(transit)
				else:
					all_transits = np.vstack([all_transits,transit])
			fname = make_plot2(datet, ra, dec, t0, per, dur, obname,dest='out/'+obname+'/')
			plots.append(fname)
