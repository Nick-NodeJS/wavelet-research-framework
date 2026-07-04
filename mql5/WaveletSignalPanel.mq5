//+------------------------------------------------------------------+
//| WaveletSignalPanel.mq5                                           |
//| MT5 Visual Signal Panel — renders market-state from Python       |
//|                                                                  |
//| Requests POST /market-state from Python Wavelet Service.         |
//| Displays:                                                        |
//|   - Trend direction                                              |
//|   - Deviation side and normalized value                          |
//|   - Historical return probability                                |
//|   - Expected bars to return                                      |
//|   - Signal: BUY / SELL / NO_TRADE / HOLD                        |
//|   - No-trade reasons                                             |
//|   - Service status and latency                                   |
//|                                                                  |
//| No calculations. Python owns all logic.                          |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "1.00"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//--- Inputs
input string InpServerUrl        = "http://127.0.0.1:5000";  // Server URL
input int    InpTickWindow       = 2048;                      // Tick Window
input int    InpRequestTimeoutMs = 500;                       // Timeout (ms)
input bool   InpAutoRefresh      = true;                      // Auto Refresh
input int    InpPanelX           = 10;                        // Panel X
input int    InpPanelY           = 20;                        // Panel Y

//--- Panel object names
#define PANEL_BG       "WvPanel_BG"
#define PANEL_STATUS   "WvPanel_Status"
#define PANEL_SIGNAL   "WvPanel_Signal"
#define PANEL_DEV      "WvPanel_Dev"
#define PANEL_PROB     "WvPanel_Prob"
#define PANEL_BARS     "WvPanel_Bars"
#define PANEL_FILTER   "WvPanel_Filter"
#define PANEL_LATENCY  "WvPanel_Latency"

//--- State
string g_last_status  = "Connecting...";
string g_last_signal  = "---";
double g_last_dev     = 0.0;
string g_last_side    = "near";
double g_last_prob    = 0.0;
double g_last_bars    = 0.0;
string g_last_filter  = "";
double g_last_latency = 0.0;
color  g_signal_color = clrGray;

//+------------------------------------------------------------------+
int OnInit()
{
   _CreatePanel();
   Comment("WaveletSignalPanel active. Server: " + InpServerUrl);
   return INIT_SUCCEEDED;
}

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
      _UpdateStatus("Connecting...", clrGray);
      return 0;
   }

   //--- Build JSON
   string json = "{\"symbol\":\"" + _Symbol + "\",\"ticks\":[";
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
   json += "]}";

   //--- POST /market-state
   char   post_data[];
   char   response_data[];
   string response_headers;

   StringToCharArray(json, post_data, 0, StringLen(json));
   ArrayResize(post_data, StringLen(json));

   string headers  = "Content-Type: application/json\r\n";
   string endpoint = InpServerUrl + "/market-state";

   int http_code = WebRequest(
      "POST", endpoint, headers, InpRequestTimeoutMs,
      post_data, response_data, response_headers
   );

   if (http_code == -1)
   {
      _UpdateStatus("SERVICE_OFFLINE", clrRed);
      _UpdatePanel();
      return 0;
   }
   if (http_code == 408)
   {
      _UpdateStatus("TIMEOUT", clrOrange);
      _UpdatePanel();
      return 0;
   }
   if (http_code != 200)
   {
      _UpdateStatus("INVALID_RESPONSE [" + IntegerToString(http_code) + "]", clrOrange);
      _UpdatePanel();
      return 0;
   }

   //--- Parse response
   string resp = CharArrayToString(response_data);

   g_last_dev     = _ExtractDouble(resp, "\"normalized\":");
   g_last_side    = _ExtractString(resp, "\"side\":\"");
   g_last_prob    = _ExtractDouble(resp, "\"return_to_trend_probability\":");
   g_last_bars    = _ExtractDouble(resp, "\"median_bars_to_return\":");
   g_last_latency = _ExtractDouble(resp, "\"latency_ms\":");
   string sig     = _ExtractString(resp, "\"side\":\"");
   // signal.side comes after filter block — use last occurrence heuristic
   int sig_pos = StringFind(resp, "\"signal\":");
   if (sig_pos >= 0)
      sig = _ExtractString(StringSubstr(resp, sig_pos, 200), "\"side\":\"");

   bool can_trade = StringFind(resp, "\"can_trade\":true") >= 0;
   string reasons = "";
   if (!can_trade)
      reasons = _ExtractArrayFirst(resp, "\"reasons\":[\"");

   g_last_signal = sig;
   g_last_filter = can_trade ? "" : reasons;
   g_last_status = "Connected";

   if (sig == "BUY")        g_signal_color = clrDodgerBlue;
   else if (sig == "SELL")  g_signal_color = clrOrangeRed;
   else                     g_signal_color = clrGray;

   if (!can_trade)          g_signal_color = clrDimGray;

   _UpdatePanel();
   return rates_total;
}

//+------------------------------------------------------------------+
void _CreatePanel()
{
   int w = 260, h = 160;
   _Rect(PANEL_BG, InpPanelX, InpPanelY, w, h, clrBlack, 180);
   _Label(PANEL_STATUS,  InpPanelX+8, InpPanelY+6,  "Connecting...",    clrGray,    9);
   _Label(PANEL_SIGNAL,  InpPanelX+8, InpPanelY+26, "Signal: ---",      clrGray,    11);
   _Label(PANEL_DEV,     InpPanelX+8, InpPanelY+50, "Dev: ---",         clrSilver,  9);
   _Label(PANEL_PROB,    InpPanelX+8, InpPanelY+66, "Prob: ---",        clrSilver,  9);
   _Label(PANEL_BARS,    InpPanelX+8, InpPanelY+82, "Bars: ---",        clrSilver,  9);
   _Label(PANEL_FILTER,  InpPanelX+8, InpPanelY+100,"Filter: pass",     clrLime,    9);
   _Label(PANEL_LATENCY, InpPanelX+8, InpPanelY+118,"Latency: ---",     clrDimGray, 8);
   ChartRedraw();
}

//+------------------------------------------------------------------+
void _UpdatePanel()
{
   string side_str = g_last_side == "above" ? "▲" : (g_last_side == "below" ? "▼" : "~");
   ObjectSetString(0, PANEL_STATUS,  OBJPROP_TEXT, "⬤ " + g_last_status);
   ObjectSetString(0, PANEL_SIGNAL,  OBJPROP_TEXT, "Signal: " + g_last_signal);
   ObjectSetInteger(0, PANEL_SIGNAL, OBJPROP_COLOR, g_signal_color);
   ObjectSetString(0, PANEL_DEV,     OBJPROP_TEXT,
      "Dev: " + side_str + " " + DoubleToString(g_last_dev, 3));
   ObjectSetString(0, PANEL_PROB,    OBJPROP_TEXT,
      "Prob: " + DoubleToString(g_last_prob * 100.0, 1) + "%");
   ObjectSetString(0, PANEL_BARS,    OBJPROP_TEXT,
      "Bars: " + DoubleToString(g_last_bars, 1));
   string filter_text = (StringLen(g_last_filter) > 0) ? "Filter: " + g_last_filter : "Filter: pass";
   color  filter_col  = (StringLen(g_last_filter) > 0) ? clrOrangeRed : clrLime;
   ObjectSetString(0, PANEL_FILTER,   OBJPROP_TEXT, filter_text);
   ObjectSetInteger(0, PANEL_FILTER,  OBJPROP_COLOR, filter_col);
   ObjectSetString(0, PANEL_LATENCY,  OBJPROP_TEXT,
      "Latency: " + DoubleToString(g_last_latency, 1) + " ms");
   ChartRedraw();
}

//+------------------------------------------------------------------+
void _UpdateStatus(string status, color col)
{
   g_last_status = status;
   ObjectSetString(0, PANEL_STATUS, OBJPROP_TEXT, "⬤ " + status);
   ObjectSetInteger(0, PANEL_STATUS, OBJPROP_COLOR, col);
   ChartRedraw();
}

//+------------------------------------------------------------------+
void _Rect(string name, int x, int y, int w, int h, color bg, int alpha)
{
   ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
}

//+------------------------------------------------------------------+
void _Label(string name, int x, int y, string text, color col, int font_size)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, name, OBJPROP_COLOR, col);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, font_size);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
}

//+------------------------------------------------------------------+
double _ExtractDouble(const string json, const string key)
{
   int pos = StringFind(json, key);
   if (pos < 0) return 0.0;
   int start = pos + StringLen(key);
   int end = start;
   while (end < StringLen(json))
   {
      string ch = StringSubstr(json, end, 1);
      if (ch == "," || ch == "}" || ch == "]") break;
      end++;
   }
   return StringToDouble(StringSubstr(json, start, end - start));
}

//+------------------------------------------------------------------+
string _ExtractString(const string json, const string key)
{
   int pos = StringFind(json, key);
   if (pos < 0) return "";
   int start = pos + StringLen(key);
   int end = StringFind(json, "\"", start);
   if (end < 0) return "";
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
string _ExtractArrayFirst(const string json, const string key)
{
   int pos = StringFind(json, key);
   if (pos < 0) return "";
   int start = pos + StringLen(key);
   int end = StringFind(json, "\"", start);
   if (end < 0) return "";
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   string names[] = {
      PANEL_BG, PANEL_STATUS, PANEL_SIGNAL, PANEL_DEV,
      PANEL_PROB, PANEL_BARS, PANEL_FILTER, PANEL_LATENCY
   };
   for (int i = 0; i < ArraySize(names); i++)
      ObjectDelete(0, names[i]);
   Comment("");
}
//+------------------------------------------------------------------+
