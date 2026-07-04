//+------------------------------------------------------------------+
//| WaveletBridgeEA.mq5                                              |
//| Wavelet Bridge Expert Advisor                                     |
//|                                                                  |
//| Responsibilities:                                                |
//|   - Own ALL HTTP communication with Python Wavelet Service       |
//|   - Execute one request per new completed bar (not every tick)   |
//|   - Write parsed results into Global Variables                   |
//|   - Perform /health checks and automatic reconnect               |
//|   - Cache last successful response across indicators              |
//|                                                                  |
//| Global Variables written (prefix: "Wv_"):                        |
//|   Wv_Status        0=Disconnected 1=Connected                    |
//|   Wv_LastUpdate    Unix timestamp of last successful response     |
//|   Wv_Latency       Round-trip latency in ms                      |
//|   Wv_LastHttpCode  Last HTTP response code                        |
//|   Wv_Trend_N       Number of values in the trend array           |
//|   Wv_Trend_{i}     Trend value at index i (0 = most recent)      |
//|   Wv_RelDev_{i}    Relative deviation at index i                 |
//|   Wv_ZScore_{i}    Z-Score at index i                            |
//|   Wv_Energy_{i}    Energy at index i                             |
//|   Wv_Noise_{i}     Noise at index i                              |
//|                                                                  |
//| Indicators read these Global Variables without any HTTP.         |
//| No trading logic. EA is a pure data bridge.                      |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "1.00"
#property description "Wavelet Bridge EA — HTTP data relay to indicators"
#property strict

//--- Inputs
input string InpServerUrl         = "http://127.0.0.1:5000";  // Server URL
input int    InpTickWindow        = 2048;                      // Tick Window (MT5 local buffer)
input int    InpRequestTimeoutMs  = 800;                       // Timeout (ms)
input int    InpHealthIntervalSec = 30;                        // Health Check Interval (s)
input bool   InpVerboseLog        = false;                     // Verbose Logging
//--- Calibration inputs (sent to Python per-request)
input string InpTrendMode         = "A2";                      // Trend Mode (A1/A2/A3/A4)
input string InpWavelet           = "db4";                     // Wavelet (db2/db4/db6/sym4/sym6/coif1)
input int    InpWaveletWindow     = 256;                       // Window (128/256/512/1024/2048)
input int    InpLevel             = 2;                         // Level (1/2/3/4)

//--- Global Variable key prefix
#define GV_PREFIX        "Wv_"
#define GV_STATUS        "Wv_Status"
#define GV_LAST_UPDATE   "Wv_LastUpdate"
#define GV_LATENCY       "Wv_Latency"
#define GV_HTTP_CODE     "Wv_LastHttpCode"
#define GV_TREND_N       "Wv_Trend_N"
#define GV_TREND_MODE    "Wv_TrendMode"

//--- Internal state
datetime g_last_bar_time    = 0;
datetime g_last_health_time = 0;
bool     g_is_connected     = false;
int      g_stored_count     = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   //--- Validate trend mode
   string mode = InpTrendMode;
   StringToUpper(mode);
   if (mode != "A1" && mode != "A2" && mode != "A3" && mode != "A4")
   {
      Print("WaveletBridgeEA: invalid InpTrendMode '", InpTrendMode, "' — must be A1/A2/A3/A4");
      return INIT_PARAMETERS_INCORRECT;
   }

   //--- Validate wavelet
   string wv = InpWavelet;
   StringToLower(wv);
   if (wv != "db2" && wv != "db4" && wv != "db6" &&
       wv != "sym4" && wv != "sym6" &&
       wv != "coif1" && wv != "coif3")
   {
      Print("WaveletBridgeEA: invalid InpWavelet '", InpWavelet, "'");
      return INIT_PARAMETERS_INCORRECT;
   }

   //--- Validate window
   if (InpWaveletWindow != 128  && InpWaveletWindow != 256  &&
       InpWaveletWindow != 512  && InpWaveletWindow != 1024 &&
       InpWaveletWindow != 2048)
   {
      Print("WaveletBridgeEA: invalid InpWaveletWindow ", InpWaveletWindow);
      return INIT_PARAMETERS_INCORRECT;
   }

   //--- Validate level
   if (InpLevel < 1 || InpLevel > 4)
   {
      Print("WaveletBridgeEA: invalid InpLevel ", InpLevel, " — must be 1-4");
      return INIT_PARAMETERS_INCORRECT;
   }

   EventSetMillisecondTimer(200);
   _WriteStatus(false, 0, 0);
   //--- Store mode as numeric: A1=1, A2=2, A3=3, A4=4
   string m = InpTrendMode;
   StringToUpper(m);
   double mode_num = (m == "A1") ? 1.0 : (m == "A3") ? 3.0 : (m == "A4") ? 4.0 : 2.0;
   GlobalVariableSet(GV_TREND_MODE, mode_num);
   Print("WaveletBridgeEA: started server=", InpServerUrl,
         " wavelet=", InpWavelet,
         " window=", InpWaveletWindow,
         " level=", InpLevel,
         " mode=", InpTrendMode);

   //--- Health check on startup
   _DoHealthCheck();

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   _WriteStatus(false, 0, 0);
   GlobalVariableSet(GV_STATUS, 0);
   Print("WaveletBridgeEA: stopped");
}

//+------------------------------------------------------------------+
void OnTimer()
{
   //--- Periodic health check
   if ((int)(TimeCurrent() - g_last_health_time) >= InpHealthIntervalSec)
      _DoHealthCheck();

   //--- Request only on new completed bar
   datetime current_bar = (datetime)SeriesInfoInteger(_Symbol, PERIOD_CURRENT, SERIES_LASTBAR_DATE);
   if (current_bar <= g_last_bar_time)
      return;

   g_last_bar_time = current_bar;
   _DoWaveletRequest();
}

//+------------------------------------------------------------------+
void OnTick()
{
   //--- Intentionally empty — all work is timer-driven (new bar check)
}

//+------------------------------------------------------------------+
//| Health check via GET /health                                      |
//+------------------------------------------------------------------+
void _DoHealthCheck()
{
   g_last_health_time = TimeCurrent();

   char   empty_data[];
   char   response_data[];
   string response_headers;
   string endpoint = InpServerUrl + "/health";

   int http_code = WebRequest(
      "GET", endpoint, "", InpRequestTimeoutMs,
      empty_data, response_data, response_headers
   );

   GlobalVariableSet(GV_HTTP_CODE, http_code);

   if (http_code == 200)
   {
      if (!g_is_connected)
      {
         g_is_connected = true;
         Print("WaveletBridgeEA: service connected");
      }
   }
   else
   {
      if (g_is_connected)
      {
         g_is_connected = false;
         GlobalVariableSet(GV_STATUS, 0);
         Print("WaveletBridgeEA: service disconnected (code=", http_code, ")");
      }
   }
}

//+------------------------------------------------------------------+
//| Main wavelet data request POST /wavelet                           |
//+------------------------------------------------------------------+
void _DoWaveletRequest()
{
   //--- Collect latest N ticks
   int n_ticks = InpTickWindow;

   MqlTick ticks_arr[];
   int copied = CopyTicks(_Symbol, ticks_arr, COPY_TICKS_ALL, 0, n_ticks);
   if (copied <= 0)
   {
      if (InpVerboseLog)
         Print("WaveletBridgeEA: no ticks available");
      return;
   }

   //--- Build JSON payload
   string json = "{\"ticks\":[";
   for (int i = 0; i < copied; i++)
   {
      double bid = ticks_arr[i].bid;
      double ask = ticks_arr[i].ask;
      double mid = (bid + ask) / 2.0;
      string ts  = IntegerToString(ticks_arr[i].time);

      json += "{\"time\":\"" + ts + "\","
           +  "\"bid\":"  + DoubleToString(bid, _Digits) + ","
           +  "\"ask\":"  + DoubleToString(ask, _Digits) + ","
           +  "\"mid\":"  + DoubleToString(mid, _Digits) + "}";
      if (i < copied - 1) json += ",";
   }
   //--- Close ticks array and add calibration fields as sibling fields
   string mode_upper = InpTrendMode;
   StringToUpper(mode_upper);
   string wv_lower = InpWavelet;
   StringToLower(wv_lower);
   json += "],"
        +  "\"trend_mode\":\"" + mode_upper + "\","
        +  "\"wavelet\":\""    + wv_lower   + "\","
        +  "\"window\":"        + IntegerToString(InpWaveletWindow) + ","
        +  "\"level\":"         + IntegerToString(InpLevel)         + "}";

   //--- Execute HTTP POST
   char   post_data[];
   char   response_data[];
   string response_headers;

   StringToCharArray(json, post_data, 0, StringLen(json));
   ArrayResize(post_data, StringLen(json));

   string headers  = "Content-Type: application/json\r\n";
   string endpoint = InpServerUrl + "/wavelet";

   ulong  t_start    = GetTickCount64();
   int    http_code  = WebRequest(
      "POST", endpoint, headers, InpRequestTimeoutMs,
      post_data, response_data, response_headers
   );
   double latency_ms = (double)(GetTickCount64() - t_start);

   GlobalVariableSet(GV_HTTP_CODE, http_code);
   GlobalVariableSet(GV_LATENCY,   latency_ms);

   if (http_code != 200)
   {
      string reason = (http_code == -1)  ? "service offline"  :
                      (http_code == 408) ? "timeout"          :
                      "http " + IntegerToString(http_code);
      if (g_is_connected)
      {
         g_is_connected = false;
         GlobalVariableSet(GV_STATUS, 0);
         Print("WaveletBridgeEA: request failed — ", reason);
      }
      return;
   }

   //--- Parse response and write Global Variables
   string resp = CharArrayToString(response_data);

   double trend_vals[];
   double rel_dev_vals[];
   double z_score_vals[];
   double energy_vals[];
   double noise_vals[];

   if (!_ExtractArray(resp, "trend",              trend_vals,   copied) ||
       !_ExtractArray(resp, "relative_deviation", rel_dev_vals, copied) ||
       !_ExtractArray(resp, "z_score",            z_score_vals, copied) ||
       !_ExtractArray(resp, "energy",             energy_vals,  copied) ||
       !_ExtractArray(resp, "noise",              noise_vals,   copied))
   {
      Print("WaveletBridgeEA: parse error — invalid response");
      return;
   }

   _WriteArraysToGlobals(trend_vals, rel_dev_vals, z_score_vals, energy_vals, noise_vals, copied);
   _WriteStatus(true, (datetime)TimeCurrent(), latency_ms);

   if (!g_is_connected)
   {
      g_is_connected = true;
      Print("WaveletBridgeEA: reconnected");
   }

   if (InpVerboseLog)
      Print("WaveletBridgeEA: ok ticks=", copied, " lat=", DoubleToString(latency_ms, 1), "ms");
}

//+------------------------------------------------------------------+
//| Write all parsed arrays into Global Variables.                    |
//|                                                                  |
//| Index convention: 0 = most recent tick (same as indicator        |
//| buffer series mode index 0).                                     |
//+------------------------------------------------------------------+
void _WriteArraysToGlobals(
   const double& trend[],
   const double& rel_dev[],
   const double& z_score[],
   const double& energy[],
   const double& noise[],
   const int     count)
{
   GlobalVariableSet(GV_TREND_N, count);
   g_stored_count = count;

   for (int i = 0; i < count; i++)
   {
      //--- Response[last] = most recent tick → GV index 0
      int src = count - 1 - i;
      string si = IntegerToString(i);

      GlobalVariableSet(GV_PREFIX + "Trend_"  + si, trend[src]);
      GlobalVariableSet(GV_PREFIX + "RelDev_" + si, rel_dev[src]);
      GlobalVariableSet(GV_PREFIX + "ZScore_" + si, z_score[src]);
      GlobalVariableSet(GV_PREFIX + "Energy_" + si, energy[src]);
      GlobalVariableSet(GV_PREFIX + "Noise_"  + si, noise[src]);
   }
}

//+------------------------------------------------------------------+
//| Write connection status Global Variables.                        |
//+------------------------------------------------------------------+
void _WriteStatus(const bool connected, const datetime last_update, const double latency_ms)
{
   GlobalVariableSet(GV_STATUS,      connected ? 1.0 : 0.0);
   GlobalVariableSet(GV_LAST_UPDATE, (double)last_update);
   if (latency_ms > 0.0)
      GlobalVariableSet(GV_LATENCY, latency_ms);
}

//+------------------------------------------------------------------+
//| Parse a named numeric array from JSON.                           |
//|                                                                  |
//| Finds "key":[...] and extracts comma-separated doubles.          |
//| Returns true on success.                                         |
//+------------------------------------------------------------------+
bool _ExtractArray(const string json, const string key, double& out[], int expected)
{
   string search = "\"" + key + "\":[";
   int pos = StringFind(json, search);
   if (pos < 0) return false;

   int start = pos + StringLen(search);
   int end   = StringFind(json, "]", start);
   if (end < 0) return false;

   string content = StringSubstr(json, start, end - start);
   if (StringLen(content) == 0) return false;

   string parts[];
   int n = StringSplit(content, ',', parts);
   if (n != expected) return false;

   ArrayResize(out, n);
   for (int i = 0; i < n; i++)
      out[i] = StringToDouble(parts[i]);

   return true;
}
//+------------------------------------------------------------------+
