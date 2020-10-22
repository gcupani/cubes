from astropy.table import Table
import numpy as np
import os
import sys

def run(airmass, pwv, moond):
    # INPUT PARAMS:
    # Airmass range [1-3]
    #airm = sim.airmass
    airm = airmass
    # Precipitable Water Vapor: as for skycalc options [0.05 0.1 0.25 0.5 1.0 1.5 2.5 3.5 5.0 7.5 10.0 20.0 30.0]
    #PWV = sim.pwv
    PWV = pwv
    # Moond day w.r.t. new moon [0 - 14]
    #moond = sim.moond
    moond = moond
    
    
    # moon phase with respect to full moon in order to conver it in moon-sun separation
    moon_FLI=moond/14
    moon_FLI_1=int(moon_FLI)
    moon_sun_sep=moon_FLI*(180/1)
    moon_sun=int(moon_sun_sep)
    
    # string conversion
    airmass_str=str(airm)
    moon_sun_str=str(moon_sun)
    pwv_str=str(PWV)
    
    # !!!!!!!!!!! With SKYCALC
    # Generation of the input txt file for calling SkyCalc tool
    with open('SkyCalc_input_NEW.txt', 'w') as fp2:
        with open('SkyCalc_input.txt') as fp:
        
            cnt=1
            line = fp.readline()
            while line:
                if cnt ==1:
                    # AIRMASS
                    # Convert the string to a list
                    line_list=list(line) 
                    # Modify the characters you want
                    line_list[19]=airmass_str[0]
                    line_list[21]=airmass_str[2]
                    # Change the list back to string, by using 'join' method of strings.
                    line_new="".join(line_list) 
                    #print("Line {}: {}".format(cnt, line_new.strip()))
                
                if cnt==5:
                    # PWV
                    # Convert the string to a list
                    line_list=list(line) 
                    # Modify the characters you want
                    if PWV < 10:
                        line_list[19]=pwv_str[0]
                        line_list[20]="."
                        line_list[21]=pwv_str[2]
                    
                    elif PWV < 100:
                        line_list[19]=pwv_str[0]
                        line_list[20]=pwv_str[1]
                        line_list[21]=pwv_str[2]
                        line_list[22]=pwv_str[3]
                    
                    # Change the list back to string, by using 'join' method of strings.
                    line_new="".join(line_list) 
                    #print("Line {}: {}".format(cnt, line_new.strip()))
                
                if cnt==8:
                    # MOON
                    # Convert the string to a list
                    line_list=list(line) 
                    # Modify the characters you want
                    if moon_sun < 10:
                        line_list[19]=moon_sun_str[0]
                        line_list[20]="."
                        line_list[21]="0"
                    
                    elif moon_sun < 100:
                        line_list[19]=moon_sun_str[0]
                        line_list[20]=moon_sun_str[1]
                    
                    elif moon_sun > 100:
                        line_list[19]=moon_sun_str[0]
                        line_list[20]=moon_sun_str[1]
                        line_list[21]=moon_sun_str[2]
                        line_list[22]="."
                        line_list[23]="0"
                    
                    # Change the list back to string, by using 'join' method of strings.
                    line_new="".join(line_list) 
                    #print("Line {}: {}".format(cnt, line_new.strip()))
                
                if cnt !=1 and cnt !=8:
                    line_new=line
                
                fp2.write(line_new)
                cnt = 1 + cnt
                line = fp.readline()
        
        fp.close()
    fp2.close()
    
    # Call Skycalc
    os.system('~/.local/bin/skycalc_cli -i SkyCalc_input_NEW.txt -o SkyCalc_input_NEW_Out.fits')
    #os.system('skycalc_cli -i SkyCalc_input_NEW.txt -o SkyCalc_input_NEW_Out.fits')
    
    
    
    # Load output file
    try:
        T=Table.read('SkyCalc_input_NEW_Out.fits')
        
        
        # Extract data
        # example of 1 value --> T_1=T[19000][0]
        T_len=len(T)-1
        Sky_lam_v=np.zeros(T_len)
        Sky_flux_v=np.zeros(T_len)
        Sky_tras_v=np.zeros(T_len)
        for i in range(0,T_len):
            # lambda extraction : conversion from um to A
            Sky_lam_v[i]=T[i][0]*10000
            # flux extraction : attention the flux is in [ph/s/m2/um/"]
            Sky_flux_v[i]=T[i][1]
            # Atm-Transmission extraction
            Sky_tras_v[i]=T[i][4]
        
        #print("Sky spectrum created and saved in SkyCalc_input_NEW_Out.fits.")
    except:
        pass
        #print("Unable to run SkyCalc.")
    
if __name__ == '__main__':
    run()
