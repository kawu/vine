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
  -- * Network
    new
  , evalBia
  , evalUni
  , ProbTyp(..)

--   -- * Opaque
--   , Opaque(..)
--   , Typ(..)
--   , newO

  -- * Transparent
  , Transparent(..)
  , Config(..)
  , inpMod
  , traMod
  , biaMod
  , uniMod
  , netErrorT
  , evalInp

  -- * Data set
  , Elem(..)
  , tokens
  , replace

  -- -- * Error
  -- , netError

--   -- * Encoding
--   , Out(..)
--   , encode
--   , decode
--   -- , rightInTwo

  -- * Explicating
  , Arc.enumerate
  , Arc.explicate
  , Arc.obfuscate
  , Arc.mask

  -- * Inference
  , treeTagGlobal
  , treeTagConstrained

  -- * Trees
  , treeConnectAll
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
-- import           GHC.Natural (naturalToInt)
-- import qualified GHC.TypeNats as Nats
-- import           GHC.TypeLits (Symbol, KnownSymbol)

-- import           System.Random (randomRIO)

-- import           Control.Monad (forM_, forM, guard)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

-- import           Data.Monoid (Sum(..))
import           Data.Monoid (Any(..))
-- import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
import qualified Data.List as List
-- import qualified Data.Foldable as F
-- import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
-- import qualified Data.Array as A
import           Data.Binary (Binary)
-- import qualified Data.Vector.Sized as VS

-- import qualified Data.Number.LogFloat as LF
-- import           Data.Number.LogFloat (LogFloat)

-- import qualified Text.Read as R
-- import qualified Text.ParserCombinators.ReadP as R

-- import qualified Data.Map.Lens as ML
-- import           Control.Lens.At (ixAt)
-- import           Control.Lens.At (ix)
-- import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
-- import qualified Data.Aeson as JSON

-- import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.)) -- , (^^?))
-- import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra        as LD

import           Net.New
import           Net.Pair
import           Net.Util
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)

import qualified Format.Cupt as Cupt

import           Graph
import           Graph.SeqTree
import qualified Net.List as NL
import qualified Net.Graph.Core as Core
import           Net.Graph.Core (Labelling(..))
import           Net.Graph.Arc
  (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph.Arc as Arc
import qualified Net.Graph.BiComp as B
import qualified Net.Graph.UniComp as U
import qualified Net.Graph.Marginals as Margs
import qualified Net.Graph.Global as Global
import qualified Net.Input as I

import           Debug.Trace (trace)


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
  , _traMod :: I.NoTrans
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


-- | Run the given uni-affine network over the given graph within the context
-- of back-propagation.
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


-- | Run the given (bi-affine) network over the given graph within the context
-- of back-propagation.
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


-- | `runUni` + `runBia` + marginals
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


----------------------------------------------
-- Global decoding
----------------------------------------------


-- | Determine the node/arc labeling which maximizes the global potential over
-- the given tree and return the resulting arc labeling.
--
-- WARNING: This function is only guaranteed to work correctly if the argument
-- graph is a tree!
--
treeTagGlobal
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labelling Bool
treeTagGlobal graph labMap nodMap =
  let (trueBest, falseBest) =
        tagSubTree
          (treeRoot graph)
          graph
          (fmap Arc.explicate labMap)
          nodMap
      best = better trueBest falseBest
   in fmap getAny (bestLab best)


-- | The function returns two `Best`s:
--
--   * The best labeling if the label of the root is `True`
--   * The best labeling if the label of the root is `False`
--
tagSubTree
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> (Best, Best)
tagSubTree root graph lmap nmap =
  (bestWith True, bestWith False)
  where
    nodePot rootVal
      | rootVal = nmap M.! root
      | otherwise = 0.0
    bestWith rootVal = addNode root rootVal (nodePot rootVal) . mconcat $ do
      child <- Graph.incoming root graph
      let arc = (child, root)
          pot arcv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=rootVal, depVal=depv}
          (true, false) = tagSubTree child graph lmap nmap
      return $ List.foldl1' better
        [ addArc arc True  (pot True  True)  true
        , addArc arc False (pot False True)  true
        , addArc arc True  (pot True  False) false
        , addArc arc False (pot False False) false ]


----------------------------------------------
-- Best (global decoding)
----------------------------------------------


-- | The best arc labeling for a given subtree.
data Best = Best
  { bestLab :: Labelling Any
    -- ^ Labelling (using `Any` guarantees that disjunction is used in case some
    -- label is accidentally assigned to a given object twice)
  , bestPot :: Double
    -- ^ Total potential
  }

instance Semigroup Best where
  Best l1 p1 <> Best l2 p2 =
    Best (l1 <> l2) (p1 + p2)

instance Monoid Best where
  mempty = Best mempty 0


-- | Impossible labeling (with infinitely poor potential)
impossible :: Best
impossible = Best mempty (read "-Infinity")


-- | Choose the better `Best`.
better :: Best -> Best -> Best
better b1 b2
  | bestPot b1 >= bestPot b2 = b1
  | otherwise = b2


-- | Add the given arc, its labeling, and the resulting potential to the given
-- `Best` structure.
addArc :: Arc -> Bool -> Double -> Best -> Best
addArc arc lab pot Best{..} = Best
  { bestLab = bestLab
      {arcLab = M.insert arc (Any lab) (arcLab bestLab)}
  , bestPot = bestPot + pot 
  }


-- | Set label of the given node in the given `Best` structure.  Similar to
-- `addArc`, but used when the potential of the node has been already accounted
-- for.
setNode :: G.Vertex -> Bool -> Best -> Best
setNode node lab best@Best{..} = best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  }


-- | Add the given node, its labeling, and the resulting potential to the given
-- `Best` structure.
addNode :: G.Vertex -> Bool -> Double -> Best -> Best
addNode node lab pot Best{..} = Best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  , bestPot = bestPot + pot
  }


----------------------------------------------
-- Constrained decoding'
----------------------------------------------


-- | Constrained version of `treeTagGlobal`
treeTagConstrained
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labelling Bool
treeTagConstrained graph labMap nodMap =
  let Best4{..} =
        tagSubTreeC'
          (treeRoot graph)
          graph
          (fmap Arc.explicate labMap)
          nodMap
      best = List.foldl1' better
        -- NOTE: `falseZeroOne` can be excluded in constrained decoding
        [true, falseZeroTrue, falseMoreTrue]
   in getAny <$> bestLab best


-- | Calculate `Best3` of the subtree.
tagSubTreeC'
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph (tree)
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> Best4
tagSubTreeC' root graph lmap nmap =
  List.foldl' (<>)
    (emptyBest4 root $ nmap M.! root)
    (map bestFor children)
  where
    children = Graph.incoming root graph
    bestFor child =
      let arc = (child, root)
          pot arcv hedv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=hedv, depVal=depv}
          Best4{..} = tagSubTreeC' child graph lmap nmap
          -- NOTE: some of the configurations below are not allowed in
          -- constrained decoding and hence are commented out.
          true' = List.foldl1' better
            [ addArc arc True  (pot True  True True)  true
            , addArc arc False (pot False True True)  true
            -- , addArc arc True  (pot True  True False) falseZeroTrue
            , addArc arc False (pot False True False) falseZeroTrue
            , addArc arc True  (pot True  True False) falseOneTrue
            -- , addArc arc False (pot False True False) falseOneTrue
            , addArc arc True  (pot True  True False) falseMoreTrue
            , addArc arc False (pot False True False) falseMoreTrue
            ]
          falseZeroTrue' = List.foldl1' better
            [ addArc arc False (pot False False True)  true
            , addArc arc False (pot False False False) falseZeroTrue
            -- , addArc arc False (pot False False False) falseOneTrue
            , addArc arc False (pot False False False) falseMoreTrue
            ]
          falseOneTrue' = List.foldl1' better
            [ addArc arc True (pot True False True)  true
            -- , addArc arc True (pot True False False) falseZeroTrue
            , addArc arc True (pot True False False) falseOneTrue
            , addArc arc True (pot True False False) falseMoreTrue
            ]
       in Best4
            { true = true'
            , falseZeroTrue = falseZeroTrue'
            , falseOneTrue  = falseOneTrue'
            , falseMoreTrue = impossible
            }


----------------------------------------------
-- Best4 (constrained decoding)
----------------------------------------------


-- | The best labeling
data Best4 = Best4
  { true          :: Best
    -- ^ The label of the root is `True`.  The root's outgoing arc can be
    -- `True` or `False.
  , falseZeroTrue :: Best
    -- ^ The label of the root is `False` and all its incoming arcs are `False`
    -- too.  The outgoing arc must be `False`.
  , falseOneTrue  :: Best
    -- ^ The label of the root is `False` and exactly one of its incoming arcs
    -- is `True`.  The outgoing arc must be `True`.
  , falseMoreTrue :: Best
    -- ^ The label of the root is `False` and more than one of its incoming
    -- arcs is `True`.  The outgoing arc can be `True` or `False.
  }

instance Semigroup Best4 where
  b1 <> b2 = Best4
    { true =
        true b1 <> true b2
    , falseZeroTrue =
        falseZeroTrue b1 <> falseZeroTrue b2
    , falseOneTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseZeroTrue b2
        ]
    , falseMoreTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseZeroTrue b2
        , falseOneTrue  b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseOneTrue  b2
        , falseMoreTrue b1 <> falseMoreTrue b2
        ]
    }


-- | Empty `Best4` for a given tree node.  Think of `mempty` with obligatory
-- vertex and potential argument.
emptyBest4 :: G.Vertex -> Double -> Best4
emptyBest4 node pot = Best4
  { true = addNode node True pot mempty
  , falseZeroTrue = addNode node False 0.0 mempty
  , falseOneTrue = impossible
  , falseMoreTrue = impossible
  }


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


----------------------------------------------
-- Error
----------------------------------------------


softMaxCrossEntropy
  :: forall s. (Reifies s W)
  => Vec8 Prob
    -- ^ Target ,,true'' distribution
  -> BVar s (Vec8 Pot)
    -- ^ Output ,,artificial'' distribution (represted by potentials)
  -> BVar s Double
softMaxCrossEntropy p0 q0 =
  softMaxCrossEntropy' (Arc.unVec p0) (BP.coerceVar q0)
--   checkNaNBP "softMaxCrossEntropy" $ negate (p `dot` LBP.vmap' log' q)
--   where
--     -- p = BP.coerceVar p0 :: BVar s (R 8)
--     p = BP.coerceVar (BP.auto p0) :: BVar s (R 8)
--     q = BP.coerceVar q0
--     -- avoid NaN when p = 0 and q = 0
--     log' x
--       | x > 0 = log x
--       | otherwise = -1.0e8


-- | Softmax + cross-entropy with manual gradient calculation.  Code adapted
-- from the backprop tutorial to use the exp-normalize trick (see
-- https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick).
softMaxCrossEntropy'
  :: (KnownNat n, Reifies s W)
  => R n
    -- ^ Target *probabilities*
  -> BVar s (R n)
    -- ^ Output *potentials*
  -> BVar s Double
softMaxCrossEntropy' x = BP.liftOp1 . BP.op1 $ \y ->
  let ymax     = LD.maxElement (LA.extract y)
      expy     = LA.dvmap (exp . (\o -> o - ymax)) y
      toty     = LD.sumElements (LA.extract expy)
      softMaxY = LA.konst (1 / toty) * expy
      smce     = negate (LA.dvmap log' softMaxY `LA.dot` x)
--    in trace ("y: " ++ show (toList y)) $
--       trace ("ymax: " ++ show ymax) $
--       trace ("expy: " ++ show (toList expy)) $
--       trace ("toty: " ++ show toty) $
--       trace ("softMaxY: " ++ show (toList softMaxY)) $
--       trace ("smce: " ++ show smce) $
   in ( smce
      , \d -> LA.konst d * (softMaxY - x)
      )
  where
    toList = LD.toList . LA.extract
    -- to avoid NaN when p_i = 0 and q_i = 0
    log' x
      | x > 0 = log x
      -- TODO: is it a proper choice of epsilon?
      | otherwise = -1.0e10


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
  => M.Map a (Vec8 Prob)
    -- ^ Target values
  -> M.Map a (BVar s (Vec8 Pot))
    -- ^ Output values
  -> BVar s Double
errorOne target output = sum $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ softMaxCrossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
  => [M.Map a (Vec8 Prob)] 
    -- ^ Targets
  -> [M.Map a (BVar s (Vec8 Pot))] 
    -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError'
  :: ( Reifies s W, KnownNat dim
     , B.BiComp dim comp
     , U.UniComp dim comp'
     )
  => Core.Version
  -> [Elem (BVar s (R dim))]
  -> BVar s comp
  -> BVar s comp'
  -> BVar s Double
netError' version dataSet net netU =
  let
    inputs = map graph dataSet
    -- outputs = map (runBoth ptyp net netU) inputs
    outputs = map (runBoth version net netU) inputs
    targets = map mkTarget dataSet
  in
    errorMany targets outputs


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


-- | Net error with `Transparent`
netErrorT
  :: (Reifies s W)
  => Config
  -> Elem (R 300)
  -> BVar s Transparent
  -> BVar s Double
netErrorT cfg x net =
  case probTyp cfg of
    Marginals ->
      netError' (version cfg) [x'] (net ^^. biaMod) (net ^^. uniMod)
    Global ->
      negate $ logLL (version cfg) [x'] (net ^^. biaMod) (net ^^. uniMod)
  where
    x' = runInp x net


----------------------------------------------
-- Log-likelihood
----------------------------------------------


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
  let labelling = Core.Labelling
        { nodLab = fmap (>0.5) (nodMap el)
        , arcLab = fmap (>0.5) (arcMap el)
        }
  return $ Global.probLog
    version
    (graph el)
    labelling
    (runBia bi $ graph el)
    (runUni uni $ graph el)
