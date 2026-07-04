//+------------------------------------------------------------------+
//| WaveletOscillator.mq5                                            |
//| MT5 Oscillator — separate window for deviation/z-score/energy   |
//|                                                                  |
//| Use alongside WaveletThinIndicator.mq5.                          |
//|                                                                  |
//| No wavelet calculations exist here.                              |
//| All computation is delegated to the Python service.              |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "1.10"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

// Buffer 0: Relative Deviation
#property indicator_label1  "Rel Deviation"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrOrangeRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

// Buffer 1: Z-Score
#property indicator_label2  "Z-Score"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrMediumPurple
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

// Buffer 2: Energy
#property indicator_label3  "Energy"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGold
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

//--- Input parameters (same as WaveletThinIndicator)
input string InpServerUrl          = "http://127.0.0.1:5000";  // Server URL
input int    InpTickWindow         = 2048;                      // Tick Window
input int    InpRequestTimeoutMs   = 500;                       // Request Timeout (ms)
input bool   InpAutoRefresh        = true;                      // Auto Refresh
input bool   InpDrawRelDeviation   = true;                      // Draw Relative Deviation
input bool   InpDrawZScore         = true;                      // Draw Z-Score
input bool   InpDrawEnergy         = true;                      // Draw Energy

//--- Indicator buffers
double BufferRelDev[];
double BufferZScore[];
double BufferEnergy[];

//--- State
string g_status = "Connecting...";

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufferRelDev, INDICATOR_DATA);
   SetIndexBuffer(1, BufferZScore, INDICATOR_DATA);
   SetIndexBuffer(2, BufferEnergy, INDICATOR_DATA);

   ArraySetAsSeries(BufferRelDev, true);
   ArraySetAsSeries(BufferZScore, true);
   ArraySetAsSeries(BufferEnergy, true);

   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpTickWindow);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, InpTickWindow);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, InpTickWindow);

   IndicatorSetString(INDICATOR_SHORTNAME, "WaveletOsc");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int       rates_total,
                const int       prev_calculated,
                const datetime& time[],
                const double&   open[],
                const double&   high[],
                const double&   low[],
                const double&   close[],
                const long&     tick_volume[],
                const long&     volume[],
                const int&      spread[])
{
   if (!InpAutoRefresh && prev_calculated > 0)
      return rates_total;

   int n_ticks = MathMin(InpTickWindow, rates_total);
   if (n_ticks < 1)
      return 0;

   MqlTick ticks_arr[];
   int copied = CopyTicks(_Symbol, ticks_arr, COPY_TICKS_ALL, 0, n_ticks);
   if (copied <= 0)
   {
      Comment("WaveletOsc: Connecting...");
      return 0;
   }

   //--- Build JSON payload with millisecond timestamps and _Digits precision
   string json = "{\"ticks\":[";
   for (int i = 0; i < copied; i++)
   {
      double bid = ticks_arr[i].bid;
      double ask = ticks_arr[i].ask;
      double mid = (bid + ask) / 2.0;
      string ts_ms = IntegerToString(ticks_arr[i].time);

      json += "{\"time\":\"" + ts_ms + "\","
           +  "\"bid\":"    + DoubleToString(bid, _Digits) + ","
           +  "\"ask\":"    + DoubleToString(ask, _Digits) + ","
           +  "\"mid\":"    + DoubleToString(mid, _Digits) + "}";
      if (i < copied - 1) json += ",";
   }
   json += "]}";

   //--- POST to service
   char   post_data[];
   char   response_data[];
   string response_headers;

   StringToCharArray(json, post_data, 0, StringLen(json));
   ArrayResize(post_data, StringLen(json));

   string headers  = "Content-Type: application/json\r\n";
   string endpoint = InpServerUrl + "/wavelet";

   int http_code = WebRequest(
      "POST", endpoint, headers, InpRequestTimeoutMs,
      post_data, response_data, response_headers
   );

   if (http_code != 200)
   {
      string s = (http_code == -1)   ? "Service Offline" :
                 (http_code == 408)  ? "Timeout"         :
                 "Error " + IntegerToString(http_code);
      Comment("WaveletOsc: " + s);
      return 0;
   }

   //--- Parse response
   string response_str = CharArrayToString(response_data);

   double rel_dev_vals[];
   double z_score_vals[];
   double energy_vals[];

   if (!_ExtractArray(response_str, "relative_deviation", rel_dev_vals, copied) ||
       !_ExtractArray(response_str, "z_score",            z_score_vals, copied) ||
       !_ExtractArray(response_str, "energy",             energy_vals,  copied))
   {
      Comment("WaveletOsc: Invalid Response");
      return 0;
   }

   //--- Write buffers
   for (int i = 0; i < copied && i < rates_total; i++)
   {
      int arr_idx = copied - 1 - i;
      if (InpDrawRelDeviation) BufferRelDev[i] = rel_dev_vals[arr_idx];
      if (InpDrawZScore)       BufferZScore[i] = z_score_vals[arr_idx];
      if (InpDrawEnergy)       BufferEnergy[i] = energy_vals[arr_idx];
   }

   Comment("WaveletOsc: Connected | ticks=" + IntegerToString(copied));
   return rates_total;
}

//+------------------------------------------------------------------+
//| Parse a named array from the JSON response string.               |
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
void OnDeinit(const int reason) { Comment(""); }
//+------------------------------------------------------------------+
