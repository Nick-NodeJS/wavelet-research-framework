//+------------------------------------------------------------------+
//| WaveletThinIndicator.mq5                                         |
//| MT5 Thin Indicator — visualization client for Wavelet Service    |
//|                                                                  |
//| Responsibilities:                                                |
//|   - Collect latest N ticks                                       |
//|   - Serialize to JSON                                            |
//|   - POST to Python Wavelet Service                               |
//|   - Parse JSON response                                          |
//|   - Write indicator buffers                                      |
//|                                                                  |
//| No wavelet calculations exist here.                              |
//| All computation is delegated to the Python service.              |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "1.00"
#property indicator_chart_window
#property indicator_buffers 5
#property indicator_plots   2

// Main chart: Trend line
#property indicator_label1  "Trend"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

// Separate window: Relative Deviation
#property indicator_separate_window
#property indicator_label2  "Rel Deviation"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrOrangeRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

//--- Input parameters (mirror IndicatorConfig)
input string InpServerUrl          = "http://127.0.0.1:5000";  // Server URL
input int    InpTickWindow         = 2048;                      // Tick Window
input int    InpRequestTimeoutMs   = 500;                       // Request Timeout (ms)
input bool   InpAutoRefresh        = true;                      // Auto Refresh
input bool   InpDrawTrend          = true;                      // Draw Trend
input bool   InpDrawRelDeviation   = true;                      // Draw Relative Deviation
input bool   InpDrawZScore         = true;                      // Draw Z-Score
input bool   InpDrawEnergy         = true;                      // Draw Energy

//--- Indicator buffers
double BufferTrend[];
double BufferRelDev[];
double BufferZScore[];
double BufferEnergy[];
double BufferNoise[];

//--- State
string g_status = "Connecting...";

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufferTrend,  INDICATOR_DATA);
   SetIndexBuffer(1, BufferRelDev, INDICATOR_DATA);
   SetIndexBuffer(2, BufferZScore, INDICATOR_DATA);
   SetIndexBuffer(3, BufferEnergy, INDICATOR_DATA);
   SetIndexBuffer(4, BufferNoise,  INDICATOR_DATA);

   ArraySetAsSeries(BufferTrend,  true);
   ArraySetAsSeries(BufferRelDev, true);
   ArraySetAsSeries(BufferZScore, true);
   ArraySetAsSeries(BufferEnergy, true);
   ArraySetAsSeries(BufferNoise,  true);

   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpTickWindow);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, InpTickWindow);

   IndicatorSetString(INDICATOR_SHORTNAME, "WaveletThin");
   Comment("Wavelet Service: " + InpServerUrl);

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

   //--- Collect latest N ticks
   int n_ticks = MathMin(InpTickWindow, rates_total);
   if (n_ticks < 1)
      return 0;

   MqlTick ticks_arr[];
   int copied = CopyTicks(_Symbol, ticks_arr, COPY_TICKS_ALL, 0, n_ticks);
   if (copied <= 0)
   {
      g_status = "Connecting...";
      Comment("Wavelet: " + g_status);
      return 0;
   }

   //--- Build JSON payload
   string json = "{\"ticks\":[";
   for (int i = 0; i < copied; i++)
   {
      double bid = ticks_arr[i].bid;
      double ask = ticks_arr[i].ask;
      double mid = (bid + ask) / 2.0;

      datetime ts  = (datetime)(ticks_arr[i].time / 1000);
      string   iso = TimeToString(ts, TIME_DATE | TIME_MINUTES | TIME_SECONDS);

      json += "{\"time\":\"" + iso + "\","
           +  "\"bid\":"    + DoubleToString(bid, 5) + ","
           +  "\"ask\":"    + DoubleToString(ask, 5) + ","
           +  "\"mid\":"    + DoubleToString(mid, 5) + "}";
      if (i < copied - 1) json += ",";
   }
   json += "]}";

   //--- POST to Python service
   char   post_data[];
   char   response_data[];
   string response_headers;

   StringToCharArray(json, post_data, 0, StringLen(json));
   ArrayResize(post_data, StringLen(json));  // trim null terminator

   string headers = "Content-Type: application/json\r\n";
   string endpoint = InpServerUrl + "/wavelet";

   int http_code = WebRequest(
      "POST",
      endpoint,
      headers,
      InpRequestTimeoutMs,
      post_data,
      response_data,
      response_headers
   );

   if (http_code == -1)
   {
      g_status = "Service Offline";
      Comment("Wavelet: " + g_status);
      return 0;
   }
   if (http_code == 408)
   {
      g_status = "Timeout";
      Comment("Wavelet: " + g_status);
      return 0;
   }
   if (http_code != 200)
   {
      g_status = "Invalid Response [" + IntegerToString(http_code) + "]";
      Comment("Wavelet: " + g_status);
      return 0;
   }

   //--- Parse JSON response
   string response_str = CharArrayToString(response_data);

   double trend_vals[];
   double rel_dev_vals[];
   double z_score_vals[];
   double energy_vals[];
   double noise_vals[];

   if (!_ParseResponseArrays(response_str, copied,
                             trend_vals, rel_dev_vals,
                             z_score_vals, energy_vals, noise_vals))
   {
      g_status = "Invalid Response";
      Comment("Wavelet: " + g_status);
      return 0;
   }

   //--- Write indicator buffers (index 0 = current bar in MT5 series)
   for (int i = 0; i < copied && i < rates_total; i++)
   {
      int buf_idx = i;
      int arr_idx = copied - 1 - i;   // response[last] = current bar

      if (InpDrawTrend)      BufferTrend[buf_idx]  = trend_vals[arr_idx];
      if (InpDrawRelDeviation) BufferRelDev[buf_idx] = rel_dev_vals[arr_idx];
      if (InpDrawZScore)     BufferZScore[buf_idx]  = z_score_vals[arr_idx];
      if (InpDrawEnergy)     BufferEnergy[buf_idx]  = energy_vals[arr_idx];
      BufferNoise[buf_idx] = noise_vals[arr_idx];
   }

   g_status = "Connected";
   Comment("Wavelet: " + g_status + " | ticks=" + IntegerToString(copied));
   return rates_total;
}

//+------------------------------------------------------------------+
//| Parse a named array from the JSON response string.               |
//|                                                                  |
//| Minimal hand-written parser — MT5 has no JSON library.           |
//| Finds "key":[...] and extracts comma-separated doubles.          |
//|                                                                  |
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
   {
      out[i] = StringToDouble(parts[i]);
   }
   return true;
}

//+------------------------------------------------------------------+
//| Parse all five arrays from the response.                         |
//+------------------------------------------------------------------+
bool _ParseResponseArrays(const string json, int expected,
                          double& trend[],
                          double& rel_dev[],
                          double& z_score[],
                          double& energy[],
                          double& noise[])
{
   if (!_ExtractArray(json, "trend",              trend,   expected)) return false;
   if (!_ExtractArray(json, "relative_deviation", rel_dev, expected)) return false;
   if (!_ExtractArray(json, "z_score",            z_score, expected)) return false;
   if (!_ExtractArray(json, "energy",             energy,  expected)) return false;
   if (!_ExtractArray(json, "noise",              noise,   expected)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Cleanup                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Comment("");
}
//+------------------------------------------------------------------+
