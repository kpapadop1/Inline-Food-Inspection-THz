function ret = startmmwstudio() 
    addpath(genpath('.\'))
    
    RSTD_DLL_Path = 'C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';
    ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
    
    if (ErrStatus ~= 30000)
        disp('Error inside Init_RSTD_Connection');
        ret = 1;
        return;
    end
    
    %Example Lua Command
    % Path for external Lua Script
    strFilename = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\Scripts\\Cascade\\Cascade_Capture.lua';
    
    Lua_String = sprintf('dofile("%s")', strFilename);
    ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
    ret = 0;
end 