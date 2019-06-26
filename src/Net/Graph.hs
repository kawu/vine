{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DeriveFunctor #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

-- To automatically derive Binary
{-# LANGUAGE DeriveAnyClass #-}


-- | Processing graphs (see `Graph`) with artificial neural networks.  This
-- module allows to model labels assigned to both arcs and nodes.
--
-- The module also implements some plumbing between the different network
-- layers: input extraction and transformation, node scoring, and arc scoring.


module Net.Graph
  ( 
  -- * Config
    Config(..)
  , ProbTyp(..)

  -- * Network
  -- ** New
  , new
  -- ** "Transparent"
  , Transparent
  , evalLoc
  , netError

  -- * Data set
  , Elem(..)

  -- * Decoding
  , Dec.treeTagGlobal
  , Dec.treeTagConstrained

  -- * Trees
  , treeConnectAll
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import qualified Data.Text as T
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import           Data.Binary (Binary)

import           Control.DeepSeq (NFData)

import           Dhall (Interpret)

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.)) -- , (^^?))
import           Numeric.LinearAlgebra.Static.Backprop (R, BVar, Reifies, W)

import           Net.New (New, new)
import           Numeric.SGD.ParamSet (ParamSet)

import qualified Format.Cupt as Cupt

import           Graph
import           Graph.SeqTree
import qualified Net.Graph.Core as Core
import           Net.Graph.Arc
  (Pot, Prob, Vec8, Out(..))
import qualified Net.Graph.Arc as Arc
import qualified Net.Graph.BiComp as B
import qualified Net.Graph.UniComp as U
import qualified Net.Graph.Marginals as Margs
import qualified Net.Graph.Global as Global
import qualified Net.Graph.Decode as Dec
import qualified Net.Graph.Error as Err
import qualified Net.Input as I

-- import           Debug.Trace (trace)


----------------------------------------------
-- Config
----------------------------------------------


-- | Typ of probabilities to employ
data ProbTyp
  = Marginals
    -- ^ Marginals
  | Global
    -- ^ Global
  deriving (Generic)

instance Interpret ProbTyp


-- | Configuration concerning the selected probability variant
data Config = Config
  { probTyp :: ProbTyp
    -- ^ Type of probability (global, marginals)
  , version :: Core.Version
    -- ^ Variant (free, constrained)
  } deriving (Generic)

instance Interpret Config


----------------------------------------------
-- Transparent
----------------------------------------------


-- | The network composed from different layers responsible for input
-- extraction, node scoring, and arc scoring.
--
-- NOTE: Should be rather named sth. like `Fixed`...
data Transparent = Transparent
  { _inpMod :: I.PosDepInp 25 25
  , _traMod :: I.Scale 1074 350
  , _uniMod :: U.UniAff 350 200
  , _biaMod :: B.BiAffMix 350 200
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

-- NOTE: alternative complex networks
--
-- data Transparent = Transparent
--   { _inpMod :: I.PosDepInp 25 25
--   -- { _inpMod :: I.RawInp
--   -- , _traMod :: I.NoTrans
--   -- , _traMod :: I.ScaleLeakyRelu 350 100
--   , _traMod :: I.Scale 350 150
--   -- , _uniMod :: U.NoUni
--   , _uniMod :: U.UniAff 150 100 :& U.PairAffLeft 150 100 :& U.PairAffRight 150 100
--   -- , _uniMod :: U.UniAff 150 100
--   -- , _biaMod :: B.BiAff 150 100
--   , _biaMod :: B.BiAffMix 150 200
--   -- , _biaMod :: B.BiAff 150 100
--   -- , _biaMod :: B.NoBi
--   } deriving (Generic, Binary, NFData, ParamSet, Backprop)
--
-- -- | Should be rather named sth. like `Fixed`...
-- data Transparent = Transparent
--   { _inpMod :: I.PosDepInp 25 25
--   , _traMod :: I.Scale 350 150
--   , _uniMod :: U.UniAff 150 150
--   , _biaMod :: B.BiAffMix 150 200
--   } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''Transparent

instance (a ~ T.Text, b ~ T.Text) => New a b Transparent where
  new xs ys = Transparent
    <$> new xs ys <*> new xs ys <*> new xs ys <*> new xs ys


----------------------------------------------
-- Network running
----------------------------------------------


-- | Run the input transformation layers.
runInp
  :: (Reifies s W)
  => Elem (R 300) 
  -> BVar s Transparent
  -> Elem (BVar s (R 350))
runInp x net =
  let toksEmbs = tokens x
      embs' = I.runTransform (net ^^. traMod)
            . I.runInput (net ^^. inpMod)
            $ toksEmbs
   in replace embs' x


-- | Run the given uni-affine network over the given graph.
runUni
  :: ( KnownNat dim
     , Reifies s W
     , U.UniComp dim comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with hidden states
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Output map with output potential values
runUni net graph = M.fromList $ do
  v <- graphNodes graph
  let x = U.runUniComp graph v net
  return (v, x)


-- | Run the given (bi-affine) network over the given graph.
runBia
  :: ( KnownNat dim --, Ord a, Show a, Ord b, Show b
     , Reifies s W
     , B.BiComp dim comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Output map with output potential values
runBia net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = B.runBiComp graph arc net
  return (arc, x)


-- | `runUni` + `runBia` + marginal scores (depending on the selected
-- `Core.Version`).
runBoth
  :: ( KnownNat dim -- , Ord a, Show a, Ord b, Show b
     , Reifies s W
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => Core.Version
  -> BVar s comp
  -> BVar s comp'
  -> Graph (BVar s (R dim)) ()
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Output map with output potential values
runBoth version net netU graph =
  case version of
    Core.Free ->
      Margs.marginalsMemo graph (runBia net graph) (runUni netU graph)
    Core.Constrained ->
      Margs.marginalsMemoC graph (runBia net graph) (runUni netU graph)
    Core.Local ->
      Margs.dummyMarginals graph (runBia net graph) (runUni netU graph)


----------------------------------------------
-- Network evaluation
----------------------------------------------


-- | Evaluate the input transformation layers.
evalInp
  :: Elem (R 300) 
  -> Transparent
  -> Elem (R 350)
evalInp x net =
  let toksEmbs = tokens x
      embs' = I.evalTransform (net ^. traMod)
            . I.evalInput (net ^. inpMod)
            $ toksEmbs
   in replace embs' x


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `runUni`.
evalUni
  :: ( KnownNat dim
     , U.UniComp dim comp
     )
  => comp
    -- ^ Graph network / params
  -> Graph (R dim) ()
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map G.Vertex Double
evalUni net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runUni
        (BP.constVar net)
        (nmap BP.auto graph)
    )


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `runBia`.
evalBia
  :: ( KnownNat dim
     , B.BiComp dim comp
     )
  => comp
    -- ^ Graph network / params
  -> Graph (R dim) ()
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (Vec8 Pot)
evalBia net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runBia
        (BP.constVar net)
        (nmap BP.auto graph)
    )


-- | Evaluate the entire network over the given element.  The result is a pair
-- of two maps with the node and arc local scores.
--
-- NOTE: The resulting scores are local, not "marginal"!  This is intentional,
-- as it allows to perform global decoding.  An alternative would be to somehow
-- decode based on marginal scores, but we didn't explore this possibility.
--
evalLoc
  :: Transparent
  -> Elem (R 300) 
  -> ( M.Map G.Vertex Double
     , M.Map Arc (Vec8 Pot)
     )
evalLoc net el0 =
  ( evalUni (net ^. uniMod) gr
  , evalBia (net ^. biaMod) gr
  )
  where
    el = evalInp el0 net
    gr = graph el


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem a = Elem
  { graph :: Graph a ()
    -- ^ Input graph
  , arcMap :: M.Map Arc Double
    -- ^ Target arc probabilities
  , nodMap :: M.Map G.Vertex Double
    -- ^ Target node probabilities
  , tokMap :: M.Map G.Vertex Cupt.Token
    -- ^ Map of .cupt tokens
  } deriving (Show, Generic, Binary)

instance Functor Elem where
  fmap f el = el {graph = nmap f (graph el)}


-- | Get the list of tokens with the corresponding labels present in the given
-- training element.
tokens :: Elem a -> [(Cupt.Token, a)]
tokens el = do
  (v, tok) <- M.toList (tokMap el)
  let y = just $ nodeLabAt (graph el) v
  return (tok, y)
  where
    just Nothing = error "Neg.Graph.tokens: got Nothing"
    just (Just x) = x


-- | Replace the labels in the given training element.
replace :: [b] -> Elem a -> Elem b
replace xs el =
  el {graph = nmap' update (graph el)}
  where
    update v _ = newMap M.! v
    newMap = M.fromList $ do
      ((v, _tok), x) <- zip (M.toList $ tokMap el) xs
      return (v, x)


-- | Create the target map from the given dataset element.
mkTarget :: Elem a -> M.Map Arc (Vec8 Prob)
mkTarget el = M.fromList $ do
  ((v, w), arcP) <- M.toList (arcMap el)
  let vP = nodMap el M.! v
      wP = nodMap el M.! w
      target = Out
        { arcVal = arcP
        , hedVal = wP
        , depVal = vP }
  return ((v, w), Arc.encode target)


----------------------------------------------
-- Error
----------------------------------------------


-- | Net error with `Transparent` over the given dataset `Elem`.  Depending on
-- the configuration, uses negated `logLL` or `crossEntropyErr`.
netError
  :: (Reifies s W)
  => Config
  -> Elem (R 300)
  -> BVar s Transparent
  -> BVar s Double
netError cfg x net =
  case probTyp cfg of
    Marginals ->
      crossEntropyErr (version cfg) [x'] (net ^^. biaMod) (net ^^. uniMod)
    Global ->
      negate $ logLL (version cfg) [x'] (net ^^. biaMod) (net ^^. uniMod)
  where
    x' = runInp x net


-- | Cross-entropy-based network error over a given dataset.
crossEntropyErr
  :: ( Reifies s W, KnownNat dim
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => Core.Version
  -> [Elem (BVar s (R dim))]
  -> BVar s comp
  -> BVar s comp'
  -> BVar s Double
crossEntropyErr version dataSet net netU =
  let
    outputs = map (runBoth version net netU) (map graph dataSet)
    targets = map mkTarget dataSet
  in
    Err.errorMany targets outputs


-- | Log-likelihood of the given dataset
logLL
  :: ( Reifies s W, KnownNat dim
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => Core.Version
  -> [Elem (BVar s (R dim))]
  -> BVar s comp
  -> BVar s comp'
  -> BVar s Double
logLL version dataSet bi uni = sum $ do
  el <- dataSet
  let labelling = Labeling
        { nodLab = fmap (>0.5) (nodMap el)
        , arcLab = fmap (>0.5) (arcMap el)
        }
  return $ Global.probLog
    version
    (graph el)
    labelling
    (runBia bi $ graph el)
    (runUni uni $ graph el)
