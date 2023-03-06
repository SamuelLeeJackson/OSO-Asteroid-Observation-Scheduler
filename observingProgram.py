#!/usr/bin/python3
from astroquery.jplhorizons import Horizons
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import SkyCoord
import argparse
import requests
import matplotlib.pyplot as plt
from pprint import pprint
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
from datetime import datetime
from datetime import timedelta
import os

def rate(x, a1, a2):
    return a1*np.square(x) + a2*x


def rate_fit(x, a1, a2, a3):
    return a1*np.square(x) + a2*x + a3
    

def get_program_rmse(start, programLength, object, loc):

    sd = start
    ed = sd + timedelta(minutes=programLength)
    startdate = sd.strftime('%Y-%m-%d %H:%M:%S')
    enddate = ed.strftime('%Y-%m-%d %H:%M:%S')
    
    obj = Horizons(id=object, location=loc, epochs={'start': startdate,
                                                       'stop': enddate,
                                                       'step': '1m'})
    ephem = obj.ephemerides(quantities='1', extra_precision=True).to_pandas()

    rateList = []
    for j in range(0, len(ephem.index)-1, 1):
        rateList.append(np.sqrt(np.square(((ephem['RA'].iloc[j+1] - ephem['RA'].iloc[j])/1)) + np.square((ephem['DEC'].iloc[j+1] - ephem['DEC'].iloc[j])/1)))
    endIndex = len(ephem.index)-1
    rateList.append(np.sqrt(np.square((ephem['RA'].iloc[endIndex] - ephem['RA'].iloc[endIndex-1])/1) + np.square((ephem['DEC'].iloc[endIndex] - ephem['DEC'].iloc[endIndex-1])/1)))

    # Degrees per minute (difference between ephemerides at 1 min intervals)
    ephem['rate'] = rateList
    
    # Arcseconds per minute
    ephem['rate'] = ephem['rate']*3600

    ephem['minutes'] = (ephem['datetime_jd'] - min(ephem['datetime_jd']))*24*60
    
    x = ephem.minutes
    y = ephem.rate
    popt, pcov = curve_fit(rate_fit, x, y)
    a1 = popt[0]
    a2 = popt[1]
    a3 = popt[2]
    mean = np.mean(y)
    residuals = y - mean
    rmse = np.sqrt(np.mean(np.square(residuals)))
    
    return rmse


def getRmse(progLen, a1, a2):
    x = np.linspace(0, progLen*60)
    y = rate(x, a1, a2)
    mean = np.mean(y)
    residuals = y - mean
    rmse = np.sqrt(np.mean(np.square(residuals)))
    
    return rmse
    
    
def getRmsePlot(progLen, a1, a2):
    x = np.linspace(0, progLen*60)
    y = rate(x, a1, a2)
    mean = np.mean(y)
    residuals = y - mean
    rmse = np.sqrt(np.mean(np.square(residuals)))
    
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14,10))
    frame1=fig.add_axes((.1,.3,.8,.6))
    ax=plt.gca()
    ax.plot(x, y)
    ax.axhline(mean, color='k', linestyle='--')
    ax.set_ylabel("Tracking Rate [arcsec/min]")
    ax.xaxis.set_tick_params(labelbottom=False, which='both', direction='in', labelsize=14)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    frame2=fig.add_axes((.1,.1,.8,.19))
    ax = plt.gca()
    ax.plot(x, residuals)
    ax.set_xlabel("Time Since Beginning of Program [minutes]")
    ax.set_ylabel("Residuals [arcsec/min]")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.axhline(0, color='k', linestyle='--')
    
    return rmse


parser = argparse.ArgumentParser(
    description='Create observing schedule for an asteroid for a given time frame, and upload the schedule.'
)

parser.add_argument(
    '-obj', '--object',
    default=None,
    help='Provide a MPC object number to observe.'
)

parser.add_argument(
    '-sd', '--startdate',
    default=None,
    help='Provide a datetime in YYYY-MM-DDThh:mm to begin the observation.'
)

parser.add_argument(
    '-ed', '--enddate',
    default=None,
    help='Provide a datetime in YYYY-MM-DDThh:mm to end the observation.'
)
parser.add_argument(
    '-exp', '--exposure',
    default='300',
    help='Provide an exposure time for the observations, default is 300 seconds.'
)
parser.add_argument(
    '-v', '--verbose',
    action='store_true',
    help='Provide additional plots for optimisation steps.'
)
parser.add_argument(
    '-p', '--priority',
    default=500,
    help='Set the program scheduling priority.'
)
parser.add_argument(
    '-t', '--telescope',
    default='pirate'
)
parser.add_argument(
    '-l', '--lum',
    default=False,
    help='Provide additional plots for optimisation steps.'
)
parser.add_argument(
    '-b', '--binning',
    default=2,
    help='Set the CCD binning mode (1x1 or 2x2).'
)

args = parser.parse_args()

if args.object is None:
    print('Object not specified, exiting.....')
    quit()
elif args.startdate is None:
    print('Start date not specified, exiting.....')
    quit()
elif args.enddate is None:
    print('End date not specified, exiting.....')
    quit()
    
if args.object == '2000KA':
    object = '2000 KA'
elif args.object == '2015NU13':
    object = '2015 NU13'
else:
    object = args.object
    
# Arguments passed to the program, or their defaults if not stated
startdate = args.startdate
enddate = args.enddate
exptime = args.exposure
verbose = args.verbose
priority = args.priority
telescope = args.telescope
lum = args.lum
binning = int(args.binning)

programDate = Time(startdate, format='isot', scale='utc')
programDate_jd = int(programDate.jd)
programDate = Time(programDate_jd, format='jd', scale='utc')
programDate = programDate.strftime('%Y%m%d')
print(f"Program base date: {programDate}")


if not os.path.isdir('Programs/'):
    os.mkdir('Programs/')
if not os.path.isdir(f'Programs/{object}/'):
    os.mkdir(f'Programs/{object}/')
if not os.path.isdir(f'Programs/{object}/{programDate}/'):
    os.mkdir(f'Programs/{object}/{programDate}/')

def get_request_data(times):
    
    start = times[0].strftime('%Y-%m-%d %H:%M:%S')
    end = times[1].strftime('%Y-%m-%d %H:%M:%S')
    obj = Horizons(id=object, location=pirate, epochs={'start': start,
                                                       'stop': end,
                                                       'step': '1m'})
    print(obj)
    ephem = obj.ephemerides(quantities='1', extra_precision=True).to_pandas()

    ephem['minutes'] = (ephem['datetime_jd'] - min(ephem['datetime_jd']))*24*60
    
    # Hours
    programLength = ephem.minutes.max()/60

    RAdiff = ephem.RA.values[-1] - ephem.RA.values[0]
    DECdiff = ephem.DEC.values[-1] - ephem.DEC.values[0]

    # Gives rate in degrees per hour which is equal to arcsec/sec
    RArate = RAdiff / programLength
    DECrate = DECdiff / programLength

    RAmid = np.mean(ephem.RA.values)
    DECmid = np.mean(ephem.DEC.values)

    coord = SkyCoord(ra=RAmid*u.degree, dec=DECmid*u.degree)
    coord = coord.to_string('hmsdms').split(' ')
    RA = coord[0].replace('h',':').replace('m',':').replace('s','')
    DEC = coord[1].replace('d',':').replace('m',':').replace('s','')

    return RA, DEC, RArate, DECrate

# Set up location for observations (PIRATE)
pirate = {'lon': -16.510297, 'lat': 28.299286, 'elevation': 2.370}

# Set up an astropy time object containing the start and end times
# Reads time in YYYY-MM-DDThh:mm:ss format
times = Time([startdate, enddate], format='isot', scale='utc')

print("Sending initial request to Horizons...")
# Get the midpoint RA, DEC, RArate & DECrate
RA, DEC, RA_rate, DEC_rate = get_request_data(times)
print("Received initial response from Horizons...")

paramsAltered = False

# Observing program length in seconds
progLen = (times[1] - times[0])
progLen = progLen.sec

# Work out the combined on-sky rate of motion (RoM)
# arcseconds per second
combinedRate = np.sqrt(np.square(RA_rate) + np.square(DEC_rate))

print("Checking exposure time trail length criterion...")
maxTrailLenPix = 200
maxTrailLenArcsec = maxTrailLenPix * 0.47
exptime = float(exptime)
trailLen = combinedRate * exptime
maxExpTime = maxTrailLenArcsec / combinedRate
if exptime > maxExpTime:
    exptime = maxExpTime
    print(f"Exposure time produces trails > 100 px... exposure time reduced to {exptime} to reduce trail length")


# FoV of PIRATE Mk.3 (June 2020) is 43 arcmin (set to 40 for margin of error)
# FoV of PIRATE Mk.4 (July 2021) us 32 arcmin (set to 30 for margin of error)
# Set up in arcseconds

if telescope.lower() == 'pirate':
    FoV = 30 * 60
else:
    FoV = 41 * 60

# Maximum program length to avoid the asteroid moving beyond the FoV
# is the FoV [arcseconds] / combined RoM [arcseconds per second]
# maxLen is therefore in units of seconds
maxLen = FoV / combinedRate

start = times[0]
end = times[1]
# If the program length is larger than the max length, then set the end time
# equal to the start time plus the maximum program length
print("Assessing program length FoV criterion...")

if progLen > maxLen:
    print("Reducing program time to avoid asteroid moving beyond FoV...")
    progLen = maxLen
    start = times[0]
    end = times[0] + TimeDelta(progLen, format='sec')
    paramsAltered = True

times = Time([start.value, end.value], format='isot', scale='utc')

# We are now dealing with minutes
min_progLen = 30
max_progLen = progLen/60
progLen_range = np.arange(min_progLen, max_progLen+10, 10)

exp_minutes = exptime / 60.
# Set maximum total tracking error to 1 arcsecond
maxTrackingError = 1 / exp_minutes

print("Assessing program length RMS tracking error criterion...")

max_progLen_rmse = get_program_rmse(start, max_progLen, object, pirate)

if max_progLen_rmse <= maxTrackingError:

    print("Maximum program length provides acceptable tracking error...")
    
else:
    print("Maximum program length does not provide acceptable tracking error...")
    print("Checking smaller program lengths to find an acceptable solution...")
    rmseArr = []
    for programLength in progLen_range:
        print(f"Calculating RMSE tracking error for {programLength} minute program length...")
        rmse = get_program_rmse(start, programLength, object, pirate)
        rmseArr.append(rmse)

    if verbose:
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(14,10))
        ax = plt.gca()
        ax.plot(progLen_range, rmseArr)
        ax.set_xlabel("Program Length [minutes]")
        ax.set_ylabel("RMS Residual [arcsec/min]")
        plt.savefig(f'Programs/{object}/{programDate}/progLenOptimisation.pdf', bbox_inches='tight')
        plt.close()

    validRMSE = [True if x <= maxTrackingError else False for x in rmseArr]
    validProgLengths = [progLen_range[i] for i in range(len(progLen_range)) if validRMSE[i]]
    maxValidProgLen = max(validProgLengths)
    print(f"Maximum allowed program length = {maxValidProgLen} minutes...")

    # Convert to minutes
    progLen = progLen/60
    if progLen > maxValidProgLen:
        print("Reducing program time to avoid RMS tracking errors > 1 arcsec...")
        start = times[0]
        end = times[0] + TimeDelta(maxValidProgLen*60, format='sec')
        times = Time([start.value, end.value], format='isot', scale='utc')
        paramsAltered = True
    
    
if paramsAltered:
    print("Program timings have been altered, requesting new ephemerides for new timings...")
    # Get the midpoint RA, DEC, RArate & DECrate
    RA, DEC, RA_rate, DEC_rate = get_request_data(times)
    # Work out the combined on-sky rate of motion (RoM)
    # arcseconds per second
    combinedRate = np.sqrt(np.square(RA_rate) + np.square(DEC_rate))

if verbose:
    print("Verbose plot generation (rates + rate params)...")
    start = times[0].strftime('%Y-%m-%d %H:%M:%S')
    end = times[1].strftime('%Y-%m-%d %H:%M:%S')
    
    obj = Horizons(id=object, location=pirate, epochs={'start': start,
                                                       'stop': end,
                                                       'step': '1m'})
                                                       
    ephem = obj.ephemerides(quantities='1', extra_precision=True).to_pandas()

    rateList = []
    for j in range(0, len(ephem.index)-1, 1):
        rateList.append(np.sqrt(np.square(((ephem['RA'].iloc[j+1] - ephem['RA'].iloc[j])/1)) + np.square((ephem['DEC'].iloc[j+1] - ephem['DEC'].iloc[j])/1)))
    endIndex = len(ephem.index)-1
    rateList.append(np.sqrt(np.square((ephem['RA'].iloc[endIndex] - ephem['RA'].iloc[endIndex-1])/1) + np.square((ephem['DEC'].iloc[endIndex] - ephem['DEC'].iloc[endIndex-1])/1)))

    ephem['rate'] = rateList
    ephem['rate'] = ephem['rate']*3600

    ephem['minutes'] = (ephem['datetime_jd'] - min(ephem['datetime_jd']))*24*60
    
    x = ephem.minutes
    y = ephem.rate
    popt, pcov = curve_fit(rate_fit, x, y)
    a1 = popt[0]
    a2 = popt[1]
    a3 = popt[2]
    mean = np.mean(y)
    residuals = y - mean
    rmse = np.sqrt(np.mean(np.square(residuals)))
    
    print("Plotting rate and residuals from mean against time...")
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14,10))
    frame1=fig.add_axes((.1,.3,.8,.6))
    ax=plt.gca()
    ax.scatter(x, y, c='k')
    xlim = ax.get_xlim()
    x_fit = np.linspace(*xlim)
    y_fit = rate_fit(x_fit, a1, a2, a3)
    ax.plot(x_fit, y_fit, c='k', linestyle='--')
    ax.axhline(mean, color='k', linestyle='--')
    ax.set_ylabel("Tracking Rate [arcsec/min]")
    ax.xaxis.set_tick_params(labelbottom=False, which='both', direction='in', labelsize=14)
    plt.xlim(xlim)
    frame2=fig.add_axes((.1,.1,.8,.19))
    ax = plt.gca()
    ax.plot(x, residuals)
    ax.set_xlabel("Time Since Beginning of Program [minutes]")
    ax.set_ylabel("Residuals [arcsec/min]")
    ax.axhline(0, color='k', linestyle='--')
    plt.savefig(f'Programs/{object}/{programDate}/rates.pdf', bbox_inches='tight')
    plt.close()
    
    programLength = ephem.minutes.max()/60

    min_a1 = -2/3600
    max_a1 = 2/3600
    a1_range = np.linspace(min_a1, max_a1, 500)

    min_a2 = -2/60
    max_a2 = 2/60
    a2_range = np.linspace(min_a2, max_a2,  500)
    
    print("Calculating contours for rate parameters plot...")

    arr = np.zeros((len(a1_range), len(a2_range)))
    for i in range(len(a1_range)):
        a1 = a1_range[i]
        for j in range(len(a2_range)):
            a2 = a2_range[j]
            arr[i,j] = getRmse(programLength, a1, a2)
            
    print("Out of contours loop...")
    xx, yy = np.meshgrid(a1_range, a2_range)
            
    plt.figure(1, (12, 10))
    plt.xlim(min(a1_range), max(a1_range))
    plt.ylim(min(a2_range), max(a2_range))
    plt.xlabel(r'$a_1$ [arcsec per minute cubed]')
    plt.ylabel(r'$a_2$ [arcsec per minute squared]')
    plt.title(f'Program Length = {programLength:.2f} hours')

    levels = [0.2, 1/3, 1]
    cs = plt.contour(xx, yy, arr.T, levels)
    fmt = {}
    strs = ['5 minutes', '3 minutes', '1 minute']
    for l, s in zip(cs.levels, strs):
        fmt[l] = s
        
    plt.clabel(cs, inline=True, fmt=fmt)
    plt.scatter(popt[0], popt[1], marker='x', color='y', s=100)
    plt.savefig(f'Programs/{object}/{programDate}/rateParams.pdf', bbox_inches='tight')
    plt.close()

# Delete this line when scheduler BST issue is resolved.
# Temporarily adds and hour to the schedule so when the scheduler
# subtracts an hour off it will be correct.
# print("Adding 1 hour shift to timings sent to scheduler (bug workaround)")
# temp_scheduler_offset = TimeDelta(3600.0, format='sec')
# times = times + temp_scheduler_offset

if lum:
    atom_str = f'Luminance:{exptime}'
else:
    atom_str = f'R:{exptime};V:{exptime}'

params = {
    'key': 'REDACTED_API_KEY',
    'name': f'A{args.object}_{programDate}',
    'observatory': f'{telescope}',
    'isTimed': 'true',
    'ra': f'{RA}',
    'dec': f'{DEC}',
    'raVel': f'{RA_rate}',
    'decVel': f'{DEC_rate}',
    'velignt': '0.00001',
    'atoms': atom_str,
    'priority': priority,
    'binning': binning,
    'altLimit': 20,
    'validFrom': f'{times.isot[0]}Z',
    'validTo': f'{times.isot[1]}Z'
}

print("Sending request to OSO scheduler as follows: ")

pprint(params)
# sending post request and saving response as response object
response = requests.post('REDACTED_API_URL/relay/addRequest', data=params)
# extracting response text
output = response.text
print(f"Your upload to the OSO API was: {output}")